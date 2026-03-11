#!/bin/bash
#SBATCH --partition=standard
#SBATCH --gres=gpu:1
#SBATCH --gpus=1 --constraint="GPUMEM80GB|GPUMEM96GB|GPUMEM140GB"
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=10:00:00
#SBATCH --output=/scratch/jhehli/logs/%x-%j.out
#SBATCH --error=/scratch/jhehli/logs/%x-%j.err

# Single nnUNet fold training job.
# Expected env vars (passed via --export):
#   FOLD  — the fold number (0-4)

set -euo pipefail

# export nnUNet dataset paths (nnUNet expects these env vars to be set)
# .bashrc sets these but SLURM jobs don't load .bashrc, so we set them here explicitly
export nnUNet_raw="/home/jhehli/data/datasets/nnUNet/nnUNet_raw"
export nnUNet_preprocessed="/home/jhehli/data/datasets/nnUNet/nnUNet_preprocessed"
export nnUNet_results="/home/jhehli/data/datasets/nnUNet/nnUNet_results"

: "${FOLD:?FOLD env var is required (0-4)}"

REPO_ROOT="/home/jhehli/data/ForkSight"
cd "$REPO_ROOT"

echo "CWD: $(pwd)"
echo "FOLD: ${FOLD}"
echo "Job ${SLURM_JOB_ID} on $(hostname)"

# load environment variables (sets NNUNET_DATASET_ID, paths, etc.)
set -a
source "${REPO_ROOT}/Environment/.env"
set +a

: "${NNUNET_DATASET_ID:?NNUNET_DATASET_ID must be set in Environment/.env}"

# activate nnUNet virtual environment
source ~/.nnUNet_env/bin/activate

mkdir -p "/scratch/jhehli/logs"

nnUNetv2_train "$NNUNET_DATASET_ID" 2d "$FOLD" -tr nnUNetTrainerWandb --npz
