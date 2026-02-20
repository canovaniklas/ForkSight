#!/bin/bash -l
# Submit a SLURM array where each task runs ONE W&B agent with --count 1.
#
# Usage (run from repo root):
#   bash Segmentation/SAM/sweep_wandb_submit_slurm.sh <entity/project/sweep_id> <n_agents>
#
# Example:
#   bash Segmentation/SAM/sweep_wandb_submit_slurm.sh EM_IMCR_BIOVSION/ForkSight-SAM/hjjdo3ya 8

set -euo pipefail

SWEEP_PATH="${1:?Usage: bash sweep_wandb_submit_slurm.sh <entity/project/sweep_id> <n_agents>}"
N_AGENTS="${2:?Usage: bash sweep_wandb_submit_slurm.sh <entity/project/sweep_id> <n_agents>}"

cd /home/jhehli/data/ForkSight

echo "Submitting ${N_AGENTS} agent(s) for sweep: ${SWEEP_PATH}"
sbatch --array=1-"${N_AGENTS}" \
  --job-name="wandb-agent-${SWEEP_PATH##*/}" \
  --export=ALL,SWEEP_PATH="${SWEEP_PATH}" \
  Segmentation/SAM/sweep_wandb_agent_job.sh

