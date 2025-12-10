#!/bin/bash -l
#SBATCH --job-name=SAM-LoRA-train
#SBATCH --partition=standard
#SBATCH --gres=gpu:1
#SBATCH --gpus=1 --constraint="GPUMEM80GB|GPUMEM96GB|GPUMEM140GB"
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=24:00:00

set -euo pipefail

# navigate to the repository root
cd "$(dirname "$0")/../.."

# load and export environment variables
set -a
source ./Segmentation/.env
set +a

# create directories for wandb and logs
mkdir -p "$WANDB_DIR" "$LOG_DIR"

# redirect stdout and stderr to log files, because env variables cannot be used in SBATCH --output and --error
exec > >(tee "$LOG_DIR/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out") \
     2> >(tee "$LOG_DIR/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.err" >&2)

echo "Logs going into: $LOG_DIR/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.*"

echo "wandb dir: $WANDB_DIR"
echo "log dir: $LOG_DIR"
echo "Job $SLURM_JOB_ID on $(hostname)"

# activate virtual environment
source "$SAM_LORA_VENV/bin/activate"

# check GPUs
srun -l bash -lc 'echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES; nvidia-smi; \
python -c "import torch; print(\"CUDA available:\", torch.cuda.is_available()); \
print(\"Device 0:\", torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\")"'

srun python -m Segmentation.SAM.sam_lora_train