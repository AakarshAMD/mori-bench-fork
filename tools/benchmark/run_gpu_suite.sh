#!/bin/bash
set -euo pipefail

: "${MORI_BENCH_REPO:?}"
: "${MORI_BENCH_SCRATCH:?}"

mkdir -p \
  "${MORI_BENCH_SCRATCH}/out" \
  "${MORI_BENCH_SCRATCH}/rocprof_p1" \
  "${MORI_BENCH_SCRATCH}/rocprof_p2" \
  "${MORI_BENCH_SCRATCH}/logs"

mapfile -t NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
[[ ${#NODES[@]} -eq 2 ]] || { echo "Expected exactly two nodes"; exit 2; }
N0="${NODES[0]}"
N1="${NODES[1]}"
MASTER="$(srun --overlap -N1 -n1 -w "$N0" hostname -I | awk '{print $1}')"

{
  echo "job_id=${SLURM_JOB_ID}"
  echo "nodes=${N0},${N1}"
  echo "master=${MASTER}"
  echo "repo=${MORI_BENCH_REPO}"
  echo "commit=$(git -C "$MORI_BENCH_REPO" rev-parse HEAD)"
  echo "remote=$(git -C "$MORI_BENCH_REPO" remote get-url origin)"
  echo "branch=$(git -C "$MORI_BENCH_REPO" branch --show-current)"
  echo "clean_status=$(git -C "$MORI_BENCH_REPO" status --porcelain | wc -l)"
  echo "image=${MORI_BENCH_IMAGE}"
  echo "fabric=MORI_RDMA_DEVICES=mlx5_0"
  echo "mem_type=gpu VRAM-to-VRAM"
  echo "MORI_ENABLE_DMABUF_REG=1"
  echo "qp_per_transfer=${MORI_BENCH_QPS:-2}"
} | tee "${MORI_BENCH_SCRATCH}/run_metadata.txt"

run_pair() {
  local name="$1" port="$2" rocp="$3" rocp_dir="$4"
  shift 4
  echo "===== ${name} port=${port} ROCPROF=${rocp} =====" | tee -a "${MORI_BENCH_SCRATCH}/driver.log"
  echo "bench_args: --mem-type gpu $*" | tee -a "${MORI_BENCH_SCRATCH}/driver.log"
  bash "${MORI_BENCH_REPO}/tools/benchmark/run_gpu_rank.sh" \
    0 "$N0" "$MASTER" "$port" "$rocp" "$rocp_dir" --mem-type gpu "$@" &
  local p0=$!
  sleep 8
  bash "${MORI_BENCH_REPO}/tools/benchmark/run_gpu_rank.sh" \
    1 "$N1" "$MASTER" "$port" "$rocp" "$rocp_dir" --mem-type gpu "$@" &
  local p1=$!
  set +e
  wait "$p0"; local e0=$?
  wait "$p1"; local e1=$?
  set -e
  echo "${name} exit rank0=${e0} rank1=${e1}" | tee -a "${MORI_BENCH_SCRATCH}/driver.log"
  [[ $e0 -eq 0 && $e1 -eq 0 ]]
}

COMMON=(
  --op-type write
  --num-initiator-dev 1
  --num-target-dev 1
  --num-worker-threads 1
  --num-qp-per-transfer "${MORI_BENCH_QPS:-2}"
  --iters 20
  --warmup 2
  --line-rate-gbps 400
  --bw-mode e2e
)

run_pair 4a_gpu_sweep 29510 0 "${MORI_BENCH_SCRATCH}/rocprof_unused" \
  "${COMMON[@]}" --transfer-batch-size 1 --all --sweep-max-size 8388608 \
  --csv "${MORI_BENCH_SCRATCH}/out/4a_summary.csv" \
  --csv-raw "${MORI_BENCH_SCRATCH}/out/4a_raw.csv"

run_pair 4b_gpu_batch8 29512 0 "${MORI_BENCH_SCRATCH}/rocprof_unused" \
  "${COMMON[@]}" --enable-batch-transfer --transfer-batch-size 8 --all \
  --sweep-start-size 1048576 --sweep-max-size 1048576 \
  --csv "${MORI_BENCH_SCRATCH}/out/4b_summary.csv" \
  --csv-raw "${MORI_BENCH_SCRATCH}/out/4b_raw.csv"

run_pair P1_gpu_profile 29514 1 "${MORI_BENCH_SCRATCH}/rocprof_p1" \
  "${COMMON[@]}" --transfer-batch-size 1 --all \
  --sweep-start-size 2097152 --sweep-max-size 2097152 \
  --csv "${MORI_BENCH_SCRATCH}/out/p1_summary.csv" \
  --csv-raw "${MORI_BENCH_SCRATCH}/out/p1_raw.csv"

run_pair P2_gpu_profile 29516 1 "${MORI_BENCH_SCRATCH}/rocprof_p2" \
  "${COMMON[@]}" --enable-batch-transfer --transfer-batch-size 8 --all \
  --sweep-start-size 1048576 --sweep-max-size 1048576 \
  --csv "${MORI_BENCH_SCRATCH}/out/p2_summary.csv" \
  --csv-raw "${MORI_BENCH_SCRATCH}/out/p2_raw.csv"

python3 "${MORI_BENCH_REPO}/tools/profiler/analyze_io_marker_trace.py" \
  "${MORI_BENCH_SCRATCH}/rocprof_p1" --line-rate-gbps 400 \
  --out-csv "${MORI_BENCH_SCRATCH}/out/p1_marker_analysis.csv" \
  --out-stripes-csv "${MORI_BENCH_SCRATCH}/out/p1_qp_stripes.csv" \
  --out-logical-csv "${MORI_BENCH_SCRATCH}/out/p1_logical_transfers.csv" \
  >"${MORI_BENCH_SCRATCH}/out/p1_marker_analysis.log"
python3 "${MORI_BENCH_REPO}/tools/profiler/analyze_io_marker_trace.py" \
  "${MORI_BENCH_SCRATCH}/rocprof_p2" --line-rate-gbps 400 \
  --out-csv "${MORI_BENCH_SCRATCH}/out/p2_marker_analysis.csv" \
  --out-stripes-csv "${MORI_BENCH_SCRATCH}/out/p2_qp_stripes.csv" \
  --out-logical-csv "${MORI_BENCH_SCRATCH}/out/p2_logical_transfers.csv" \
  >"${MORI_BENCH_SCRATCH}/out/p2_marker_analysis.log"

python3 "${MORI_BENCH_REPO}/tools/benchmark/validate_gpu_artifacts.py" \
  "${MORI_BENCH_SCRATCH}" | tee "${MORI_BENCH_SCRATCH}/validation.log"
echo "ALL_PHASES_COMPLETE job=${SLURM_JOB_ID}" | tee -a "${MORI_BENCH_SCRATCH}/driver.log"
