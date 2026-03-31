#!/bin/bash
#SBATCH --job-name=OneAstro-bench
#SBATCH --account=root
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --output=/dev/null
#SBATCH --exclusive

source ~/bin/packages/miniconda3/etc/profile.d/conda.sh
conda activate AION

# Recommended NCCL settings for Alps (adjust according to CSCS docs if needed)
export OMP_NUM_THREADS=8
export NCCL_DEBUG=WARN
unset NCCL_BLOCKING_WAIT
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_DISTRIBUTED_DEBUG=OFF

# slurm config
NNODES="${SLURM_JOB_NUM_NODES}"
GPUS_PER_NODE="${SLURM_GPUS_PER_NODE:-4}"
NODE_RANK="${SLURM_NODEID}"
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
MASTER_PORT="${MASTER_PORT:-29500}"

# log file 
DIR='/capstor/store/cscs/pasc/c39/swiss-ai/benchmark'
# DIR='./reports'
LOG_DIR="${DIR}/${SLURM_JOB_ID}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}-${NNODES}N.out"
GSSR_OUTPUT="${LOG_DIR}/gssr_report"

# Get the directory of the current script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
2>&1 | tee "$LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

# Model and cache config
MODEL_NAME=${MODEL_NAME:-"polymathic-ai/aion-base"}

# Benchmark config
WARMUP_STEPS="${WARMUP_STEPS:-200}"
MEASURE_STEPS="${MEASURE_STEPS:-1800}"
# WARMUP_STEPS="${WARMUP_STEPS:-100}"
# MEASURE_STEPS="${MEASURE_STEPS:-200}"
BASELINE_GPUS=4
BASELINE_BATCH_SIZE=256
GLOBAL_BATCH_TARGET="${GLOBAL_BATCH_TARGET:-$((BASELINE_GPUS * BASELINE_BATCH_SIZE))}"
DEFAULT_BATCH_SIZE=$((GLOBAL_BATCH_TARGET / (NNODES * GPUS_PER_NODE)))
if (( DEFAULT_BATCH_SIZE < 1 )); then
  echo "Error: computed per-GPU batch size ${DEFAULT_BATCH_SIZE} is invalid for NNODES=${NNODES}, GPUS_PER_NODE=${GPUS_PER_NODE}, GLOBAL_BATCH_TARGET=${GLOBAL_BATCH_TARGET}" >&2
  exit 1
fi
if (( GLOBAL_BATCH_TARGET % (NNODES * GPUS_PER_NODE) != 0 )); then
  echo "Error: GLOBAL_BATCH_TARGET=${GLOBAL_BATCH_TARGET} is not divisible by total GPUs=$((NNODES * GPUS_PER_NODE))" >&2
  exit 1
fi
BATCH_SIZE="${BATCH_SIZE:-$DEFAULT_BATCH_SIZE}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
NUM_WORKERS="${NUM_WORKERS:-16}"
DATA_PATH="${DATA_PATH:-/capstor/store/cscs/pasc/c39/swiss-ai/test/data/train/}"

# Tokenizer and model config
INPUT_BUDGET=${INPUT_BUDGET:-256}
ANCHOR_RATIO_MIN=${ANCHOR_RATIO_MIN:-0.8}
ANCHOR_RATIO_MAX=${ANCHOR_RATIO_MAX:-1.6}
OUTPUT_BUDGET=${OUTPUT_BUDGET:-128}
BETA_ALPHA=${BETA_ALPHA:-1.0}
BETA_BETA=${BETA_BETA:-4}

echo "Benchmark launcher config:"
echo "  nnodes=$NNODES"
echo "  gpus_per_node=$GPUS_PER_NODE"
echo "  global_batch_target=$GLOBAL_BATCH_TARGET"
echo "  batch_size_per_gpu=$BATCH_SIZE"
echo "  grad_accum_steps=$GRAD_ACCUM"
echo "  num_workers=$NUM_WORKERS"
echo "  input_budget=$INPUT_BUDGET"
echo "  anchor_ratio_min=$ANCHOR_RATIO_MIN"
echo "  anchor_ratio_max=$ANCHOR_RATIO_MAX"
echo "  output_budget=$OUTPUT_BUDGET"
echo "  beta_alpha=$BETA_ALPHA"
echo "  beta_beta=$BETA_BETA"
echo "  warmup_steps=$WARMUP_STEPS"
echo "  measure_steps=$MEASURE_STEPS"
echo "  data_path=$DATA_PATH"

GSSR=~/bin/packages/GPU-Saturation-Scorer/gssr-record
srun "$GSSR" -o "$GSSR_OUTPUT" torchrun \
  --nnodes="$NNODES" \
  --node_rank="$NODE_RANK" \
  --nproc_per_node="$GPUS_PER_NODE" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
  ./train_aion_ddp_v4.12_token.py \
  --model_name "$MODEL_NAME" \
  --batch_size "$BATCH_SIZE" \
  --weight_decay 0.1 \
  --num_workers 16 \
  --use_amp \
  --use_tensorboard \
  --grad_accum_steps "$GRAD_ACCUM" \
  --num_workers "$NUM_WORKERS" \
  --input_budget "$INPUT_BUDGET" \
  --anchor_ratio_min "$ANCHOR_RATIO_MIN" \
  --anchor_ratio_max "$ANCHOR_RATIO_MAX" \
  --output_budget "$OUTPUT_BUDGET" \
  --beta_alpha "$BETA_ALPHA" \
  --beta_beta "$BETA_BETA" \
  --data_path "$DATA_PATH" \
  --output_dir "${LOG_DIR}" \
  --benchmark_only \
  --benchmark_warmup_steps "$WARMUP_STEPS" \
  --benchmark_measure_steps "$MEASURE_STEPS" 
