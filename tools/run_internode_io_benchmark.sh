#!/bin/bash
# Run one node of the two-node MORI-IO RDMA benchmark via torchrun.
#
# Usage:
#   run_internode_io_benchmark.sh \
#     --rank <0|1> \
#     --master-addr <ip-or-hostname> \
#     --ifname <nic> \
#     [--master-port <port>] \
#     [--host <io-engine-host>] \
#     -- [benchmark.py args...]
#
# The benchmark args after `--` are forwarded to tests/python/io/benchmark.py.
# The script always runs the RDMA backend in 2-node mode with nproc_per_node=1.
# Timeout can be overridden via MORI_IO_BENCH_TIMEOUT_SEC.
#
# ------------------------------------------------------------------------------
# OPT-IN rocprofv3 marker-trace capture (default OFF; behavior byte-identical when
# unset). Set ROCPROF=1 to:
#   * export MORI_ROCTX=1 and MORI_ROCTX_TRANSFER=1 so the MoRI-IO host RDMA path
#     emits its roctx ranges (mori.rdma.kv_transfer[.read], mori.rdma.batch_post.*,
#     mori.io.engine_batch_write). These gates are OFF unless ROCPROF=1.
#   * wrap torchrun with:
#       rocprofv3 --marker-trace --kernel-trace \
#                 --output-format pftrace csv json \
#                 -d <ROCPROF_OUTDIR> -o %hostname%_%pid% --
#
# Why marker-trace (not just kernel-trace): a MoRI RDMA write/read is a HOST-side
# ibv_post_send + NIC hardware op, NOT a HIP kernel, so --kernel-trace is BLIND to
# the transfer path (empty for --mem-type cpu). The MORI_ROCTX_TRANSFER async
# ranges (post -> CQE) are the real wire-duration lens; analyze them with
# tools/profiler/analyze_io_marker_trace.py.
#
# Tunables (only read when ROCPROF=1):
#   ROCPROF_OUTDIR   output dir for traces (default: ./rocprof_io_out)
#   ROCPROF_BIN      rocprofv3 binary (default: rocprofv3 on PATH)
#   ROCPROF_EXTRA    extra args spliced before the `--` (e.g. "--stats")
#
# Known ROCm-7.2 rocprofiler-sdk quirks to watch for (report if seen):
#   * async post/CQ marker pattern can emit non-fatal correlation_id.cpp WARN/ERROR
#     spam -- benign.
#   * a SIGINT/SIGTERM re-entrant finalize deadlock can yield 0 output files if the
#     wrapped process is signaled instead of exiting normally (e.g. `timeout` firing).
#     The benchmark should exit normally; if the outdir is empty, suspect this.

set -euo pipefail

RANK=""
MASTER_ADDR=""
MASTER_PORT=1234
IFNAME=""
HOST=""
NUMA_NODE=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --rank)         RANK="$2";        shift 2 ;;
    --master-addr)  MASTER_ADDR="$2"; shift 2 ;;
    --master-port)  MASTER_PORT="$2"; shift 2 ;;
    --ifname)       IFNAME="$2";      shift 2 ;;
    --host)         HOST="$2";        shift 2 ;;
    --numa)         NUMA_NODE="$2";   shift 2 ;;
    --)             shift; EXTRA_ARGS=("$@"); break ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

for var in RANK MASTER_ADDR IFNAME; do
  [[ -z "${!var}" ]] && { echo "Missing required argument for --${var,,}"; exit 1; }
done

if [[ -z "$HOST" ]]; then
  HOST="$(
    python3 - "$IFNAME" <<'PY'
import fcntl
import socket
import struct
import sys

ifname = sys.argv[1].encode("utf-8")
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    packed = struct.pack("256s", ifname[:15])
    addr = fcntl.ioctl(sock.fileno(), 0x8915, packed)[20:24]
    print(socket.inet_ntoa(addr))
except OSError:
    pass
PY
  )"
fi

if [[ -z "$HOST" ]]; then
  echo "Failed to determine local host address for interface '$IFNAME'; pass --host explicitly" >&2
  exit 1
fi

export GLOO_SOCKET_IFNAME="$IFNAME"
export MORI_SOCKET_IFNAME="$IFNAME"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_ROOT"

BENCH_TIMEOUT_SEC="${MORI_IO_BENCH_TIMEOUT_SEC:-600}"

# Optional NUMA pinning. For host-memory multi-NIC runs this is REQUIRED to stay
# rail-safe: the multi-NIC pairing matches each side's rank-j NUMA-local NIC, so
# both nodes must select the SAME NIC subset. Pinning both nodes to the same NUMA
# node makes MatchCpuNics() return an identical, rail-aligned NIC ordering on both
# ends; without it the two nodes can land on different NUMA nodes and pair NICs
# across rails (fails on fabrics where rails are not interconnected).
NUMACTL=()
if [[ -n "$NUMA_NODE" ]]; then
  if command -v numactl >/dev/null 2>&1; then
    NUMACTL=(numactl --cpunodebind="$NUMA_NODE" --membind="$NUMA_NODE")
    echo "[run_internode_io_benchmark] NUMA pinned to node $NUMA_NODE: ${NUMACTL[*]}"
  else
    echo "[run_internode_io_benchmark] ERROR: --numa $NUMA_NODE requested but numactl not found;" \
         "refusing to run multi-NIC host benchmark unpinned (cross-rail risk)." >&2
    exit 1
  fi
fi

# Optional rocprofv3 marker-trace wrapper (opt-in via ROCPROF=1). When OFF this
# array is empty and the exec line is identical to the original.
ROCPROF_PREFIX=()
if [[ "${ROCPROF:-0}" == "1" || "${ROCPROF:-}" =~ ^[tTyY] ]]; then
  # Turn ON the MoRI-IO roctx gates ONLY for this profiled run.
  export MORI_ROCTX=1
  export MORI_ROCTX_TRANSFER=1
  ROCPROF_BIN="${ROCPROF_BIN:-rocprofv3}"
  ROCPROF_OUTDIR="${ROCPROF_OUTDIR:-$REPO_ROOT/rocprof_io_out}"
  mkdir -p "$ROCPROF_OUTDIR"
  read -r -a _rocprof_extra <<< "${ROCPROF_EXTRA:-}"
  ROCPROF_PREFIX=(
    "$ROCPROF_BIN"
    --marker-trace
    --kernel-trace
    --output-format pftrace csv json
    -d "$ROCPROF_OUTDIR"
    -o "%hostname%_%pid%"
    "${_rocprof_extra[@]}"
    --
  )
  echo "[run_internode_io_benchmark] ROCPROF=1: rank $RANK capturing marker+kernel trace"
  echo "[run_internode_io_benchmark]   MORI_ROCTX=1 MORI_ROCTX_TRANSFER=1"
  echo "[run_internode_io_benchmark]   outdir: $ROCPROF_OUTDIR  bin: $ROCPROF_BIN"
  echo "[run_internode_io_benchmark]   files:  %hostname%_%pid%.{pftrace,csv,json}"
  if ! command -v "$ROCPROF_BIN" >/dev/null 2>&1; then
    echo "[run_internode_io_benchmark] WARNING: '$ROCPROF_BIN' not found on PATH;" \
         "the run will likely fail. Install/point ROCPROF_BIN at rocprofv3." >&2
  fi
fi

exec "${NUMACTL[@]}" timeout "$BENCH_TIMEOUT_SEC" "${ROCPROF_PREFIX[@]}" torchrun \
  --nnodes=2 \
  --node_rank="$RANK" \
  --nproc_per_node=1 \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  -m tests.python.io.benchmark \
  --backend rdma \
  --host "$HOST" \
  "${EXTRA_ARGS[@]}"
