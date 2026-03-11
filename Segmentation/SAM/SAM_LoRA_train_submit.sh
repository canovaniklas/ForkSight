#!/bin/bash -l
#SBATCH --job-name=SAM-LoRA-train
#SBATCH --partition=standard
#SBATCH --gres=gpu:1
#SBATCH --gpus=1 --constraint="GPUMEM80GB|GPUMEM96GB|GPUMEM140GB"
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --output=/home/jhehli/scratch/logs/%x-%j.out
#SBATCH --error=/home/jhehli/scratch/logs/%x-%j.err

set -euo pipefail

# navigate to the repository root
cd /home/jhehli/data/ForkSight
echo "Current working directory: $(pwd)"

# load and export environment variables
set -a
source ./Environment/.env
set +a

# apply any env var overrides passed as arguments (e.g. SAM_LORA_LR=0.0001)
for arg in "$@"; do
    export "$arg"
done

# create directories for wandb and logs
BASE_DIR="/scratch/jhehli"
export WANDB_DIR="${BASE_DIR}/wandb"
export LOG_DIR="${BASE_DIR}/logs"

mkdir -p "$WANDB_DIR" "$LOG_DIR"

echo "wandb dir: $WANDB_DIR"
echo "log dir: $LOG_DIR"
echo "Job $SLURM_JOB_ID on $(hostname)"

# activate virtual environment
source "$SAM_LORA_VENV/bin/activate"

# check GPUs
srun -l bash -lc 'echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES; nvidia-smi; \
python -c "import torch; print(\"CUDA available:\", torch.cuda.is_available()); \
print(\"Device 0:\", torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\")"'

# create GPU log file, start GPU logging in the background
#LOG_FILE="$LOG_DIR/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_gpu_usage.csv"
#echo "Time,GPU,Name,MemTotal,MemUsed,MemFree,UtilGPU" > "$LOG_FILE"

# 
#(
#  while true; do
#    TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")
#    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
#               --format=csv,noheader,nounits | while read -r line; do
#      echo "$TIMESTAMP,$line" >> "$LOG_FILE"
#    done
#    sleep 30
#  done
#) &

# run training script
srun python -u -m Segmentation.SAM.sam_lora_train