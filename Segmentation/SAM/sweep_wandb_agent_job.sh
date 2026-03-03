#!/bin/bash
#SBATCH --partition=standard
#SBATCH --gres=gpu:1
#SBATCH --gpus=1 --constraint="GPUMEM80GB|GPUMEM96GB|GPUMEM140GB"
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=10:00:00
#SBATCH --output=/scratch/jhehli/logs/%x-%A_%a.out
#SBATCH --error=/scratch/jhehli/logs/%x-%A_%a.err

# One SLURM task runs one W&B agent which executes exactly one sweep trial (--count 1).
# Called via sweep_wandb_submit_slurm.sh:
#   sbatch --array=1-N --export=ALL,SWEEP_PATH=entity/project/sweep_id \
#       Segmentation/SAM/sweep_wandb_agent_job.sh

set -euo pipefail

: "${SWEEP_PATH:?SWEEP_PATH env var is required (entity/project/sweep_id)}"

# go to repo root
cd /home/jhehli/data/ForkSight
echo "CWD: $(pwd)"
echo "SWEEP_PATH: ${SWEEP_PATH}"
echo "Job ${SLURM_JOB_ID}, array task ${SLURM_ARRAY_TASK_ID:-n/a} on $(hostname)"

# load base environment variables (.env sets SAM_LORA_VENV, WANDB_API_KEY, paths, etc.)
set -a
source ./Segmentation/.env
set +a

# activate virtual environment (SAM_LORA_VENV is set by .env above)
source "$SAM_LORA_VENV/bin/activate"

# W&B dirs on scratch (fast, not backed up — suitable for run artifacts)
export WANDB_DIR="/scratch/jhehli/wandb"
export WANDB_CACHE_DIR="/scratch/jhehli/wandb_cache"
export WANDB_CONFIG_DIR="/scratch/jhehli/wandb_config"
export WANDB_SILENT="true"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" "/scratch/jhehli/logs"

# wandb agent writes sweep param files under ./wandb/sweep-<id>/
SWEEP_ID="${SWEEP_PATH##*/}"
mkdir -p "wandb/sweep-${SWEEP_ID}"

# run exactly one sweep trial on the allocated GPU and exit
srun wandb agent --count 1 "${SWEEP_PATH}"
