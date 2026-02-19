#!/bin/bash -l

# Launch a W&B sweep agent on the login node.
# The agent itself is lightweight — each trial submits its own SLURM
# job with a GPU, so the agent just coordinates.
#
# Usage (run from repo root):
#   bash Segmentation/SAM/sweep_wandb_submit.sh <entity/project/sweep_id> [count]
#
# Run inside tmux/screen so it survives SSH disconnects:
#   tmux new -s sweep-cldice
#   bash Segmentation/SAM/sweep_wandb_submit.sh EM_IMCR_BIOVSION/ForkSight-SAM/abc12345
#
# To limit the number of trials:
#   bash Segmentation/SAM/sweep_wandb_submit.sh EM_IMCR_BIOVSION/ForkSight-SAM/abc12345 20

set -euo pipefail

SWEEP_PATH="${1:?Usage: bash sweep_wandb_submit.sh <entity/project/sweep_id> [count]}"
COUNT="${2:-}"

cd /home/jhehli/data/ForkSight
echo "Current working directory: $(pwd)"

# load environment variables
set -a
source ./Segmentation/.env
set +a

# directories
BASE_DIR="/scratch/jhehli"
export WANDB_DIR="${BASE_DIR}/wandb"
export LOG_DIR="${BASE_DIR}/logs"
mkdir -p "$WANDB_DIR" "$LOG_DIR"

echo "wandb dir: $WANDB_DIR"
echo "log dir:   $LOG_DIR"
echo "sweep:     $SWEEP_PATH"

# activate virtual environment
source "$SAM_LORA_VENV/bin/activate"

# launch the wandb agent (each trial submits its own sbatch job)
COUNT_FLAG=""
if [ -n "$COUNT" ]; then
    COUNT_FLAG="--count $COUNT"
fi

wandb agent $COUNT_FLAG "$SWEEP_PATH"
