#!/bin/bash
# Creates one isolated uv virtual environment per third-party repo (light work: downloads + installs).
# Each venv lives inside external_code/<name>/.venv; the repos themselves are never modified.
set -euo pipefail
E=/autofs/space/crater_001/projects/micropath/semi/external_code
export UV_CACHE_DIR=$E/.uv_cache UV_LINK_MODE=hardlink UV_CONCURRENT_DOWNLOADS=4 UV_CONCURRENT_INSTALLS=2
PY=3.10
CU121=https://download.pytorch.org/whl/cu121
COMMON="numpy==1.26.4 scipy==1.11.4 scikit-learn==1.4.2 scikit-image==0.22.0 opencv-python-headless==4.10.0.84 pandas==2.2.3 matplotlib==3.8.4 tqdm pillow==10.4.0 pyyaml"

mk() { local name=$1; shift; [ -x $E/$name/.venv/bin/python ] || uv venv -q --python $PY $E/$name/.venv;
       uv pip install -q --python $E/$name/.venv/bin/python --index-strategy unsafe-best-match --extra-index-url $CU121 "$@"; echo "[ok] $name"; }

case "${1:-all}" in
  dinomaly|all)   mk dinomaly torch==2.5.1+cu121 torchvision==0.20.1+cu121 timm==0.9.12 tabulate ptflops==0.7.4 $COMMON ;;&
  inp_former|all) mk inp_former torch==2.5.1+cu121 torchvision==0.20.1+cu121 timm==0.9.12 tabulate kornia==0.7.3 adeval==1.1.0 $COMMON ;;&
  glass|all)      mk glass torch==2.1.2+cu121 torchvision==0.16.2+cu121 timm==0.9.12 imgaug==0.4.0 click==8.1.7 tensorboard==2.15.1 openpyxl==3.0.10 \
                     numpy==1.26.3 scipy==1.11.4 scikit-learn==1.4.0 scikit-image==0.22.0 opencv-python-headless==4.10.0.84 pandas==1.5.3 matplotlib==3.8.2 tqdm pillow==10.2.0 ;;&
  subspacead|all) mk subspacead torch==2.5.1+cu121 torchvision==0.20.1+cu121 "transformers>=4.48,<5" kornia $COMMON ;;&
  anomalydino|all) mk anomalydino torch==2.5.1+cu121 torchvision==0.20.1+cu121 faiss-gpu-cu12 tifffile $COMMON ;;
esac
