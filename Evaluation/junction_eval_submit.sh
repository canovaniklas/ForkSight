#!/bin/bash
# Submit junction detection evaluation as a 3-job SLURM chain.
# Jobs run in series: nnunet inference → sam inference → metrics computation.
#
# Usage (from repo root):
#   bash Evaluation/junction_eval_submit.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_SCRIPT="${SCRIPT_DIR}/junction_eval_job.sh"

echo "Submitting junction evaluation pipeline..."
echo "Job script: ${JOB_SCRIPT}"
echo ""

# --- Stage 1: nnUNet inference ---
NNUNET_JOB=$(sbatch --parsable \
    --job-name="junc-infer-nnunet" \
    --export=STAGE=nnunet \
    "$JOB_SCRIPT")
echo "Stage 1 (nnunet inference): job ${NNUNET_JOB}"

# --- Stage 2: SAM inference (waits for nnUNet) ---
SAM_JOB=$(sbatch --parsable \
    --job-name="junc-infer-sam" \
    --export=STAGE=sam \
    --dependency=afterok:"${NNUNET_JOB}" \
    "$JOB_SCRIPT")
echo "Stage 2 (sam inference):   job ${SAM_JOB} (waits for job ${NNUNET_JOB})"

# --- Stage 3: metrics computation (waits for SAM) ---
METRICS_JOB=$(sbatch --parsable \
    --job-name="junc-metrics" \
    --gres="" --gpus=0 \
    --constraint="" \
    --export=STAGE=metrics \
    --dependency=afterok:"${SAM_JOB}" \
    "$JOB_SCRIPT")
echo "Stage 3 (metrics):         job ${METRICS_JOB} (waits for job ${SAM_JOB})"

echo ""
echo "All jobs submitted. Monitor with:  squeue -u \$USER"
