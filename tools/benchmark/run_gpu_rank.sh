#!/bin/bash
set -euo pipefail

RANK="$1"
NODE="$2"
MASTER="$3"
PORT="$4"
ROCPROF_FLAG="$5"
ROCPROF_DIR="$6"
shift 6

: "${MORI_BENCH_REPO:?Set MORI_BENCH_REPO to the clean checkout}"
: "${MORI_BENCH_SCRATCH:?Set MORI_BENCH_SCRATCH to the job artifact directory}"
: "${MORI_BENCH_IMAGE:?Set MORI_BENCH_IMAGE to the runtime image}"

LOG="${MORI_BENCH_SCRATCH}/logs/rank${RANK}_port${PORT}.log"
mkdir -p "$(dirname "$LOG")"
: >"$LOG"

srun --overlap -N1 --ntasks=1 -w "$NODE" \
  docker run --rm --network host --ipc host --privileged --cap-add SYS_PTRACE \
    --ulimit memlock=-1:-1 \
    --device /dev/kfd --device /dev/dri --device /dev/infiniband \
    -v /lib/modules:/lib/modules \
    -v /sys/class/infiniband:/sys/class/infiniband:ro \
    -v /sys/class/net:/sys/class/net:ro \
    -v /shared_inference:/shared_inference \
    -e SLURM_JOB_ID="$SLURM_JOB_ID" \
    -e MORI_RDMA_DEVICES=mlx5_0 \
    -e MORI_ENABLE_DMABUF_REG=1 \
    -e ROCPROF="$ROCPROF_FLAG" \
    -e ROCPROF_OUTDIR="$ROCPROF_DIR" \
    -e MORI_IO_BENCH_TIMEOUT_SEC="${MORI_IO_BENCH_TIMEOUT_SEC:-900}" \
    --entrypoint bash "$MORI_BENCH_IMAGE" -lc '
      set -euo pipefail
      repo="$1"; rank="$2"; master="$3"; port="$4"; shift 4
      export PATH=/opt/venv/bin:$PATH

      # Build in node-local storage. The only input source is the verified clean,
      # pushed checkout mounted at $repo; no other MORI tree is on PYTHONPATH.
      build_src="/tmp/mori-bench-fork-${SLURM_JOB_ID}-${rank}"
      rm -rf "$build_src"
      cp -a "$repo" "$build_src"
      cd "$build_src"
      test -z "$(git status --porcelain)"
      python3 -m pip install --no-cache-dir -r tools/benchmark/requirements.txt
      python3 -m pip install --no-cache-dir --no-build-isolation --no-deps --force-reinstall .

      unset PYTHONPATH
      python3 - <<PY
import glob
import mori
import mori.cpp
import mori.io
print("MORI_IMPORT_FILE=" + mori.__file__)
print("MORI_CPP_FILE=" + mori.cpp.__file__)
print("MORI_IO_FILE=" + mori.io.__file__)
native = sorted(glob.glob("/opt/venv/lib/python*/site-packages/mori/**/*.so", recursive=True))
print("MORI_NATIVE_SOS=" + ":".join(native))
assert native, "no installed MORI native shared object found"
assert all("/mori_bench_run" not in p and "/mori-fork" not in p for p in native)
PY
      echo "SOURCE_REPO=$repo"
      echo "SOURCE_COMMIT=$(git -C "$repo" rev-parse HEAD)"
      echo "MORI_RDMA_DEVICES=$MORI_RDMA_DEVICES"
      echo "MORI_ENABLE_DMABUF_REG=$MORI_ENABLE_DMABUF_REG"
      exec bash "$build_src/tools/run_internode_io_benchmark.sh" \
        --rank "$rank" --master-addr "$master" --master-port "$port" --ifname eth0 -- "$@"
    ' bash "$MORI_BENCH_REPO" "$RANK" "$MASTER" "$PORT" "$@" >>"$LOG" 2>&1
