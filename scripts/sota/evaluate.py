#!/usr/bin/env python
"""Identical metrics for every SOTA run, from its saved image scores and 256x256 maps.

Run in the semimil venv (needs scripts/localization.py::pro_auc).  Settings via env vars
(MLSC sbatch does not forward arguments):
    RUN_DIRS  colon-separated run directories (each with image_scores.csv + maps256.npy)

Metrics (full real test split, 1,272 normal + 116 defective images):
    image AUROC / AUPRC; pixel AUROC (+AUPRC) over all test pixels at 256x256;
    PRO integrated to 30 % FPR (localization.pro_auc, maps at 256x256);
    recall at the 99th percentile of the test-normal image scores (1 % FPR), overall and
    per defect-size bin (area % of the 512x512 mask: <0.1, 0.1-1, >1; as analyze_size.py).
Writes metrics.json next to the inputs.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
from localization import MAP, pro_auc  # noqa: E402

MASK_DIR = Path("/autofs/space/crater_001/projects/micropath/semi/MIIC/Anomaly_test/abnormal_mask")
BINS = [(0.0, 0.1, "small"), (0.1, 1.0, "medium"), (1.0, 100.0, "large")]  # as analyze_size.py


def masks_and_areas(names: list[str], labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    masks = np.zeros((len(names), MAP, MAP), np.uint8)
    areas = np.full(len(names), np.nan)
    for i, (n, y) in enumerate(zip(names, labels)):
        if y == 0:
            continue
        m = Image.open(MASK_DIR / f"{Path(n).stem}_mask.jpg").convert("L")
        areas[i] = (np.asarray(m) > 127).mean() * 100                      # % of the 512x512 image
        masks[i] = (np.asarray(m.resize((MAP, MAP), Image.NEAREST)) > 127)  # as localization.load_mask
    return masks, areas


def evaluate(d: Path) -> dict:
    rows = list(csv.DictReader((d / "image_scores.csv").open()))
    names = [r["name"] for r in rows]
    y = np.array([int(r["label"]) for r in rows])
    s = np.array([float(r["score"]) for r in rows])
    maps = np.load(d / "maps256.npy").astype(np.float32)
    assert maps.shape == (len(rows), MAP, MAP), maps.shape
    if not os.environ.get("SMOKE"):
        assert (y == 0).sum() == 1272 and (y == 1).sum() == 116, "not the full test split"
    assert np.isfinite(s).all() and np.isfinite(maps).all()
    masks, areas = masks_and_areas(names, y)

    thr = float(np.percentile(s[y == 0], 99))
    det = s >= thr
    out = {
        "run_dir": str(d),
        "image_auroc": float(roc_auc_score(y, s)),
        "image_auprc": float(average_precision_score(y, s)),
        "pixel_auroc": float(roc_auc_score(masks.ravel(), maps.ravel())),
        "pixel_auprc": float(average_precision_score(masks.ravel(), maps.ravel())),
        "pro": pro_auc(maps, masks),
        "thr_1pct_fpr": thr,
        "fpr_at_thr": float(det[y == 0].mean()),
        "recall_1pct_fpr": float(det[y == 1].mean()),
    }
    for lo, hi, name in BINS:
        sel = (y == 1) & (areas >= lo) & (areas < hi)
        out[f"n_{name}"] = int(sel.sum())
        out[f"recall_{name}"] = float(det[sel].mean())
    info = d / "run_info.json"
    if info.exists():
        out["run_info"] = json.loads(info.read_text())
    return out


def main() -> None:
    dirs = [Path(p) for p in os.environ["RUN_DIRS"].split(":") if p]
    for d in dirs:
        t0 = time.time()
        m = evaluate(d)
        (d / "metrics.json").write_text(json.dumps(m, indent=2))
        print(f"{d}: I-AUROC {m['image_auroc']:.4f} I-AUPRC {m['image_auprc']:.4f} "
              f"P-AUROC {m['pixel_auroc']:.4f} PRO {m['pro']:.4f} R@1%FPR {m['recall_1pct_fpr']:.3f} "
              f"S/M/L {m['recall_small']:.3f}/{m['recall_medium']:.3f}/{m['recall_large']:.3f} "
              f"({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
