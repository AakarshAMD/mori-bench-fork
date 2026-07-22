#!/bin/bash
#SBATCH --job-name=mori-bench-fork-gpu
#SBATCH --partition=amd-rccl
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:mi300x:8
#SBATCH --exclusive
#SBATCH --exclude=useocpm2m-097-119
#SBATCH --time=04:00:00
#SBATCH --output=/shared_inference/aarai/distinf/mori_bench_fork_runs/slurm-%j.out

set -euo pipefail

REPO="${1:?Usage: sbatch run_mi300x_gpu_benchmark.sbatch CLEAN_CHECKOUT [EXPECTED_COMMIT]}"
EXPECTED_COMMIT="${2:-}"
IMAGE="rocmshared/pytorch-private:sglang_mori_roctx_cq_bytes_rocm720_20260702"
RUN_ROOT="${MORI_BENCH_RUN_ROOT:-/shared_inference/aarai/distinf/mori_bench_fork_runs}"
SCRATCH="${RUN_ROOT}/job_${SLURM_JOB_ID}"

mkdir -p "$SCRATCH"
exec > >(tee "$SCRATCH/slurm_${SLURM_JOB_ID}.out") 2>&1

test -d "$REPO/.git"
ACTUAL_COMMIT="$(git -C "$REPO" rev-parse HEAD)"
[[ -z "$EXPECTED_COMMIT" || "$ACTUAL_COMMIT" == "$EXPECTED_COMMIT" ]] || {
  echo "Commit mismatch: expected=$EXPECTED_COMMIT actual=$ACTUAL_COMMIT"
  exit 2
}
[[ "$(git -C "$REPO" remote get-url origin)" == *"AakarshAMD/mori-bench-fork.git"* ]]
[[ -z "$(git -C "$REPO" status --porcelain)" ]] || {
  echo "Refusing non-clean checkout:"
  git -C "$REPO" status --short
  exit 2
}
git -C "$REPO" show -s --format='CLEAN_SOURCE_COMMIT=%H%nCLEAN_SOURCE_TREE=%T%nCLEAN_SOURCE_SUBJECT=%s'

# Docker operations are intentionally inside this allocated job. Pull once on
# each allocated compute node before any build or benchmark container starts.
srun --overlap -N2 -n2 --ntasks-per-node=1 docker pull "$IMAGE"

export MORI_BENCH_REPO="$REPO"
export MORI_BENCH_SCRATCH="$SCRATCH"
export MORI_BENCH_IMAGE="$IMAGE"
bash "$REPO/tools/benchmark/run_gpu_suite.sh"
