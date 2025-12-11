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
source ./Segmentation/.env
set +a

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

srun python -u -m Segmentation.SAM.sam_lora_train