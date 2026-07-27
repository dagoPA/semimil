#!/usr/bin/env python
"""Per-defect-size analysis of the one-class detector (DINOv2), from saved scores.

Bins the real abnormal test images by defect area and reports, per bin, the
detection rate (recall) at the 1% FPR operating point and the AUROC vs. all
normals. Shows that small defects are the hard ones.

    uv run python scripts/analyze_size.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score

ROOT = Path("/autofs/space/crater_001/projects/micropath/semi")
RAW = ROOT / "MIIC"
OUT = ROOT / "outputs/patchcore/dinov2"
SCORES = OUT / "image_scores.csv"
MASK_DIR = RAW / "Anomaly_test/abnormal_mask"

# defect-area bins (% of the 512x512 image)
BINS = [(0.0, 0.1, "pequeño (<0.1%)"), (0.1, 1.0, "mediano (0.1-1%)"), (1.0, 100.0, "grande (>1%)")]


def main() -> None:
    rows = list(csv.DictReader(SCORES.open()))
    normal = np.array([float(r["score"]) for r in rows if r["label"] == "0"])
    thr = float(np.percentile(normal, 99))

    abn = [r for r in rows if r["label"] == "1"]
    areas, scores = [], []
    for r in abn:
        stem = Path(r["name"]).stem
        m = np.asarray(Image.open(MASK_DIR / f"{stem}_mask.jpg").convert("L")) > 127
        areas.append(m.mean() * 100)
        scores.append(float(r["score"]))
    areas, scores = np.array(areas), np.array(scores)

    print(f"threshold (1% FPR) = {thr:.2f}   n_abnormal={len(abn)}   n_normal={len(normal)}\n")
    print(f"{'bin':<20}{'n':>4}{'recall':>9}{'AUROC':>9}{'mean_area%':>12}")
    table = []
    for lo, hi, name in BINS:
        sel = (areas >= lo) & (areas < hi)
        n = int(sel.sum())
        if n == 0:
            continue
        recall = float((scores[sel] >= thr).mean())
        y = np.concatenate([np.zeros(len(normal)), np.ones(n)])
        s = np.concatenate([normal, scores[sel]])
        auroc = float(roc_auc_score(y, s))
        print(f"{name:<20}{n:>4}{recall:>9.3f}{auroc:>9.3f}{areas[sel].mean():>12.3f}")
        table.append({"bin": name, "n": n, "recall": recall, "auroc": auroc,
                      "mean_area_pct": float(areas[sel].mean())})

    (OUT / "per_size.json").write_text(json.dumps({"threshold": thr, "bins": table}, indent=2))
    print(f"\nsaved -> {OUT / 'per_size.json'}")


if __name__ == "__main__":
    main()
