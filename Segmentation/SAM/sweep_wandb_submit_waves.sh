#!/bin/bash -l
# Submit W&B sweep trials in waves with SLURM dependencies.
# Wave 1 (exploration) runs more jobs in parallel; subsequent waves
# run fewer so the Bayesian optimizer can learn between them.
#
# Usage (run from repo root):
#   bash Segmentation/SAM/sweep_wandb_submit_waves.sh <entity/project/sweep_id> <wave1_size> <later_wave_size> <num_later_waves>
#
# Example (10 exploration + 5 waves of 2 = 20 total trials):
#   bash Segmentation/SAM/sweep_wandb_submit_waves.sh EM_IMCR_BIOVSION/ForkSight-SAM/abc123xy 10 2 5

set -euo pipefail

SWEEP_PATH="${1:?Usage: $0 <sweep_path> <wave1_size> <later_wave_size> <num_later_waves>}"
WAVE1_SIZE="${2:?Usage: $0 <sweep_path> <wave1_size> <later_wave_size> <num_later_waves>}"
LATER_WAVE_SIZE="${3:?Usage: $0 <sweep_path> <wave1_size> <later_wave_size> <num_later_waves>}"
NUM_LATER_WAVES="${4:?Usage: $0 <sweep_path> <wave1_size> <later_wave_size> <num_later_waves>}"

SWEEP_ID="${SWEEP_PATH##*/}"
TOTAL=$(( WAVE1_SIZE + LATER_WAVE_SIZE * NUM_LATER_WAVES ))

cd /home/jhehli/data/ForkSight

echo "Sweep: ${SWEEP_PATH}"
echo "Wave 1 (exploration): ${WAVE1_SIZE} parallel trials"
echo "Waves 2-$(( NUM_LATER_WAVES + 1 )): ${NUM_LATER_WAVES} waves × ${LATER_WAVE_SIZE} parallel trials"
echo "Total trials: ${TOTAL}"
echo ""

# Wave 1: exploration — high parallelism
PREV_JOB=$(sbatch --parsable \
    --array=1-"${WAVE1_SIZE}" \
    --job-name="wandb-w1-${SWEEP_ID}" \
    --export=ALL,SWEEP_PATH="${SWEEP_PATH}" \
    Segmentation/SAM/sweep_wandb_agent_job.sh)

echo "Wave 1 (exploration): job ${PREV_JOB} — ${WAVE1_SIZE} parallel trials"

# Waves 2+: exploitation — lower parallelism, each waits for previous wave
for wave in $(seq 2 $(( NUM_LATER_WAVES + 1 ))); do
    PREV_JOB=$(sbatch --parsable \
        --array=1-"${LATER_WAVE_SIZE}" \
        --job-name="wandb-w${wave}-${SWEEP_ID}" \
        --export=ALL,SWEEP_PATH="${SWEEP_PATH}" \
        --dependency=afterany:"${PREV_JOB}" \
        Segmentation/SAM/sweep_wandb_agent_job.sh)

    echo "Wave ${wave}: job ${PREV_JOB} — ${LATER_WAVE_SIZE} parallel trials (waits for job ${PREV_JOB%_*})"
done

echo ""
echo "Monitor at: https://wandb.ai/${SWEEP_PATH%/*}/sweeps"
