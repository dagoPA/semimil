#!/bin/bash
#SBATCH --job-name=semimil
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

# Runs one argument-free experiment script on an MLSC GPU node.
# MLSC's sbatch does not forward script arguments, so everything comes from env vars:
#   SCRIPT          script under scripts/ (required), e.g. run_tiled.py
#   SEMIMIL_CONFIG  optional YAML that overrides the script's default config
#
# Launch from the repository root on mlsc:
#   SCRIPT=run_tiled.py SEMIMIL_CONFIG=configs/tiled_smoke.yaml sbatch --job-name=tiled_smoke scripts/slurm/run_script.sh

cd /homes/8/dp003/code/development/semi
: "${SCRIPT:?set SCRIPT, e.g. SCRIPT=run_tiled.py}"
export PYTHONPATH=src PYTHONUNBUFFERED=1 OMP_NUM_THREADS=3
echo "host=$(hostname) job=$SLURM_JOB_ID script=$SCRIPT config=${SEMIMIL_CONFIG:-default} start=$(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# Call the venv directly: `uv run` would try to sync the environment on the node.
.venv/bin/python "scripts/$SCRIPT"
echo "exit=$? end=$(date)"
