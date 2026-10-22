#!/bin/bash
#SBATCH --job-name=semimil_seq
#SBATCH --account=qtim
#SBATCH --partition=dgx-a100,rtx8000,rtx6000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Runs one argument-free script over several configs, one after the other, on a single
# GPU (kinder to the shared queue than one job per config). Env vars:
#   SCRIPT   script under scripts/, e.g. run_tiled.py
#   CONFIGS  space-separated YAML paths, each passed as SEMIMIL_CONFIG
cd /homes/8/dp003/code/development/semi
: "${SCRIPT:?set SCRIPT}" "${CONFIGS:?set CONFIGS}"
export PYTHONPATH=src PYTHONUNBUFFERED=1 OMP_NUM_THREADS=3
echo "host=$(hostname) job=$SLURM_JOB_ID script=$SCRIPT start=$(date)"
for c in $CONFIGS; do
  echo "===== $c $(date)"
  SEMIMIL_CONFIG="$c" .venv/bin/python "scripts/$SCRIPT" || echo "!! $c failed"
done
echo "end=$(date)"
