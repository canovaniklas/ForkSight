#!/bin/bash -l
#SBATCH --job-name=segmentation-eval
#SBATCH --partition=standard
#SBATCH --gres=gpu:1
#SBATCH --gpus=1 --constraint="GPUMEM80GB|GPUMEM96GB|GPUMEM140GB"
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/jhehli/logs/%x-%j.out
#SBATCH --error=/scratch/jhehli/logs/%x-%j.err

set -euo pipefail

cd /home/jhehli/data/ForkSight
echo "CWD: $(pwd)"
echo "Job ${SLURM_JOB_ID} on $(hostname)"

set -a
source ./Environment/.env
set +a

source "$SAM_LORA_VENV/bin/activate"

python -u -m Evaluation.compute_metrics_segmentation \
    --dataset Segmentation_v1 \
    --batch-size 8 \
    --plot
