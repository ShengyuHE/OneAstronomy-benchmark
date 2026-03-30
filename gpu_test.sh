#!/bin/bash
#SBATCH --job-name=OneAstro-bench
#SBATCH --account=root
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --output=/dev/null

source ~/bin/packages/miniconda3/etc/profile.d/conda.sh
conda activate AION

# Recommended NCCL settings for Alps (adjust according to CSCS docs if needed)
export OMP_NUM_THREADS=8
export NCCL_DEBUG=WARN
export TORCH_DISTRIBUTED_DEBUG=OFF
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# slurm config
NNODES="${SLURM_JOB_NUM_NODES}"
GPUS_PER_NODE="${SLURM_GPUS_PER_NODE:-4}"
NODE_RANK="${SLURM_NODEID}"
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
MASTER_PORT="${MASTER_PORT:-29500}"

# log file 
DIR='/capstor/store/cscs/pasc/c39/swiss-ai/test/reports'
# DIR='./reports'
LOG_DIR="${DIR}/${SLURM_JOB_ID}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}-${NNODES}N.out"
GSSR_OUTPUT="${LOG_DIR}/gssr_report"
exec > >(tee -a "$LOG_FILE") 2>&1

# Benchmark config
WARMUP_STEPS="${WARMUP_STEPS:-100}"
MEASURE_STEPS="${MEASURE_STEPS:-500}"
BATCH_SIZE="${BATCH_SIZE:-96}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
INPUT_BUDGET="${INPUT_BUDGET:-512}"
OUTPUT_BUDGET="${OUTPUT_BUDGET:-256}"
DATA_PATH="${DATA_PATH:-/capstor/store/cscs/pasc/c39/swiss-ai/test/data/train/}"

echo "Benchmark launcher config:"
echo "  nnodes=$NNODES"
echo "  gpus_per_node=$GPUS_PER_NODE"
echo "  batch_size_per_gpu=$BATCH_SIZE"
echo "  grad_accum_steps=$GRAD_ACCUM"
echo "  input_budget=$INPUT_BUDGET"
echo "  output_budget=$OUTPUT_BUDGET"
echo "  warmup_steps=$WARMUP_STEPS"
echo "  measure_steps=$MEASURE_STEPS"

GSSR=~/bin/packages/GPU-Saturation-Scorer/gssr-record
srun "$GSSR" -o "$GSSR_OUTPUT" torchrun \
  --nnodes="$NNODES" \
  --node_rank="$NODE_RANK" \
  --nproc_per_node="$GPUS_PER_NODE" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
  ./AION/train_aion_ddp_token.py \
  --batch_size "$BATCH_SIZE" \
  --grad_accum_steps "$GRAD_ACCUM" \
  --input_budget "$INPUT_BUDGET" \
  --output_budget "$OUTPUT_BUDGET" \
  --data_path "$DATA_PATH" \
  --output_dir "${LOG_DIR}" \
  --benchmark_only \
  --benchmark_warmup_steps "$WARMUP_STEPS" \
  --benchmark_measure_steps "$MEASURE_STEPS" 
