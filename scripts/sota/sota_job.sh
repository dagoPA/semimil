#!/bin/bash
#SBATCH --job-name=sota
#SBATCH --account=qtim
#SBATCH --partition=dgx-a100,rtx8000,rtx6000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --output=/autofs/space/crater_001/projects/micropath/semi/outputs/sota/logs/%x_%j.out
#SBATCH --error=/autofs/space/crater_001/projects/micropath/semi/outputs/sota/logs/%x_%j.err

# One SOTA method run on an MLSC GPU node, then the shared evaluation (scripts/sota/evaluate.py).
# MLSC's sbatch does not forward script arguments, so everything comes from env vars:
#   METHODS   space-separated list of dinomaly | inp_former | glass | subspacead | anomalydino
#   STAGE=1   copy the training normals to the node-local $TMPDIR first (Dinomaly)
#   plus the wrapper's own vars (SEED, MODE, SHOTS, SEEDS, N_TRAIN, SMOKE, ...)
# Example (from the semi repo on mlsc):
#   METHODS=dinomaly SEED=0 STAGE=1 sbatch --job-name=sota_dinomaly_s0 scripts/sota/sota_job.sh

E=/autofs/space/crater_001/projects/micropath/semi/external_code
S=/homes/8/dp003/code/development/semi/scripts/sota
OUT=/autofs/space/crater_001/projects/micropath/semi/outputs/sota
: "${METHODS:?set METHODS, e.g. METHODS=dinomaly}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
export TORCH_HOME=$E/work/torch_home HF_HOME=$E/work/hf_home HF_HUB_OFFLINE=1
# TLS roots for downloads on the nodes (as run_patchcore.py does with certifi)
export SSL_CERT_FILE=/homes/8/dp003/code/development/semi/.venv/lib/python3.11/site-packages/certifi/cacert.pem REQUESTS_CA_BUNDLE=/homes/8/dp003/code/development/semi/.venv/lib/python3.11/site-packages/certifi/cacert.pem
export RUNDIR_LIST=$OUT/logs/rundirs_${SLURM_JOB_ID}.txt
echo "host=$(hostname) job=$SLURM_JOB_ID methods=$METHODS seed=${SEED:-} smoke=${SMOKE:-0} start=$(date)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

if [ "${STAGE:-0}" = "1" ]; then
  t0=$(date +%s); mkdir -p "$TMPDIR/miic"
  cp -rL $E/miic_mvtec/miic/train "$TMPDIR/miic/"
  export TRAIN_ROOT=$TMPDIR
  echo "staged $(ls $TMPDIR/miic/train/good | wc -l) train images to $TMPDIR in $(( $(date +%s) - t0 ))s"
fi

status=0
for M in $METHODS; do
  echo "=== $M $(date)"
  $E/$M/.venv/bin/python $S/run_$M.py || { status=$?; echo "!!! $M failed with $status"; }
done

if [ -s "$RUNDIR_LIST" ]; then
  echo "=== evaluate $(date)"
  cd /homes/8/dp003/code/development/semi && \
    RUN_DIRS=$(paste -sd: "$RUNDIR_LIST") PYTHONPATH=src .venv/bin/python scripts/sota/evaluate.py || status=$?
fi
echo "exit=$status end=$(date)"
exit $status
