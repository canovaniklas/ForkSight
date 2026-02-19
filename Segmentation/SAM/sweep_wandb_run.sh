#!/bin/bash -l
#SBATCH --partition=standard
#SBATCH --gres=gpu:1
#SBATCH --gpus=1 --constraint="GPUMEM80GB|GPUMEM96GB|GPUMEM140GB"
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --output=/home/jhehli/scratch/logs/%x-%j.out
#SBATCH --error=/home/jhehli/scratch/logs/%x-%j.err

# Runs a single W&B sweep trial inside a SLURM job.
#
# Usage (called by sweep_wandb_train.py, not directly):
#   sbatch --wait sweep_wandb_run.sh <entity> <project> <run_id> <sweep_id> <params_file>

set -euo pipefail

ENTITY="$1"
PROJECT="$2"
RUN_ID="$3"
SWEEP_ID="$4"
PARAMS_FILE="$5"

cd /home/jhehli/data/ForkSight
echo "Current working directory: $(pwd)"

# load base environment variables
set -a
source ./Segmentation/.env
set +a

# apply sweep hyperparameter overrides
source "$PARAMS_FILE"

# directories
BASE_DIR="/scratch/jhehli"
export WANDB_DIR="${BASE_DIR}/wandb"
export LOG_DIR="${BASE_DIR}/logs"
mkdir -p "$WANDB_DIR" "$LOG_DIR"

echo "wandb dir:  $WANDB_DIR"
echo "log dir:    $LOG_DIR"
echo "run_id:     $RUN_ID"
echo "sweep_id:   $SWEEP_ID"
echo "params:     $PARAMS_FILE"
echo "Job $SLURM_JOB_ID on $(hostname)"

# activate virtual environment
source "$SAM_LORA_VENV/bin/activate"

# run the training worker (resumes the wandb sweep run)
srun python -u -m Segmentation.SAM.sweep_wandb_worker \
    --entity "$ENTITY" \
    --project "$PROJECT" \
    --run-id "$RUN_ID" \
    --sweep-id "$SWEEP_ID"
