#!/bin/bash
#SBATCH --job-name=day4_dist_tf
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --partition=cpu
#SBATCH --output=slurm_tf_%j.out

set -euo pipefail
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$SCRIPT_DIR/train_slurm_tf.py}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"
: "${ONEAPI_SETVARS:?Set ONEAPI_SETVARS to the Intel oneAPI setvars.sh path}"

# Temporarily disable -e while sourcing setvars.sh to handle warning return code (exit 3) gracefully
set +e
source "$ONEAPI_SETVARS" --force >/dev/null 2>&1 || true
set -e

command -v icx >/dev/null || { echo "Intel C++ compiler (icx) is unavailable after sourcing oneAPI" >&2; exit 1; }
command -v vtune >/dev/null || { echo "Intel VTune is unavailable after sourcing oneAPI" >&2; exit 1; }
command -v srun >/dev/null || { echo "Slurm srun is unavailable" >&2; exit 1; }
command -v scontrol >/dev/null || { echo "Slurm scontrol is unavailable" >&2; exit 1; }

export TF_ENABLE_ONEDNN_OPTS="${TF_ENABLE_ONEDNN_OPTS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$SLURM_CPUS_PER_TASK}"
export TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS:-$OMP_NUM_THREADS}"
export TF_NUM_INTEROP_THREADS="${TF_NUM_INTEROP_THREADS:-1}"
export KMP_AFFINITY="${KMP_AFFINITY:-granularity=fine,compact,1,0}"
export KMP_BLOCKTIME="${KMP_BLOCKTIME:-1}"

NODES=($(scontrol show hostnames "$SLURM_NODELIST"))
BASE_PORT="${TF_PORT_BASE:-$($PYTHON_BIN -c \"import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()\")}"
ENDPOINTS=""
for index in "${!NODES[@]}"; do
    ENDPOINTS="${ENDPOINTS}${NODES[$index]}:$((BASE_PORT + index)),"
done
export SLURM_TF_ENDPOINTS="${ENDPOINTS%,}"

srun "$PYTHON_BIN" "$TRAIN_SCRIPT" --omp_threads="$OMP_NUM_THREADS" --per_worker_batch="${PER_WORKER_BATCH:-64}"