#!/usr/bin/env python
"""Final MIIC metrics of the proposed detector (and its parts) from saved exhaustive-bank maps.

Proposed detector: DINOv2 block 8 tokens and block 10 tokens with a 3x3 local average,
each scored against an exhaustive memory bank (every token of the training normals),
standardized on the 200 calibration normals and averaged (image score and map).

Reads outputs/bigbank*/raw/n<images>.npz (written by run_bigbank.py) and reports, for
the 4,000-image and full banks: image AUROC/AUPRC, recall at the 99th percentile of
the test normals (overall and per size bin, with per-bin AUROC), pixel AUROC and PRO
on 256x256 maps.  Also reports the paper's 100k-token baseline threshold (mean over
the three run_tiled seeds) for the threshold table.  Writes outputs/final/metrics.json.

    uv run python scripts/analysis/final_metrics.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from localization import pro_auc  # noqa: E402
from run_patchcore import list_images  # noqa: E402

ROOT = Path("/autofs/space/crater_001/projects/micropath/semi")
RAW = ROOT / "MIIC"
OUT = ROOT / "outputs/final"
RES = 256
BINS = [(0.0, 0.1, "small"), (0.1, 1.0, "medium"), (1.0, 100.0, "large")]


def load(run: str, n: int):
    z = np.load(ROOT / f"outputs/{run}/raw/n{n}.npz")
    return z["test_map"].astype(np.float32), z["calib_map"].astype(np.float32)


def standardize(test, calib):
    """Image score = max over tokens; scores and maps standardized on calibration normals."""
    ti, ci = test.reshape(len(test), -1).max(1), calib.reshape(len(calib), -1).max(1)
    return (ti - ci.mean()) / ci.std(), (test - calib.mean()) / calib.std()


def evaluate(score, maps, y, n_norm, areas, masks):
    normal, abn = score[:n_norm], score[n_norm:]
    thr = float(np.percentile(normal, 99))
    up = F.interpolate(torch.from_numpy(maps)[:, None], size=(RES, RES), mode="bilinear",
                       align_corners=False)[:, 0].numpy()
    r = {"image_auroc": float(roc_auc_score(y, score)),
         "image_auprc": float(average_precision_score(y, score)),
         "recall_at_1fpr": float((abn >= thr).mean()),
         "pixel_auroc": float(roc_auc_score(masks.reshape(-1), up.reshape(-1))),
         "pro": float(pro_auc(up, masks))}
    for lo, hi, name in BINS:
        sel = (areas >= lo) & (areas < hi)
        r[f"recall_{name}"] = float((abn[sel] >= thr).mean())
        r[f"auroc_{name}"] = float(roc_auc_score(np.r_[np.zeros(n_norm), np.ones(sel.sum())],
                                                 np.r_[normal, abn[sel]]))
    return r


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    test_norm = list_images(sorted((RAW / "Anomaly_test/normal_img").iterdir()))
    test_abn = list_images(sorted((RAW / "Anomaly_test/abnormal_img").iterdir()))
    n_norm = len(test_norm)
    y = np.r_[np.zeros(n_norm), np.ones(len(test_abn))]
    mpaths = [RAW / "Anomaly_test/abnormal_mask" / f"{p.stem}_mask.jpg" for p in test_abn]
    areas = np.array([(np.asarray(Image.open(m).convert("L")) > 127).mean() * 100 for m in mpaths])
    masks = np.concatenate([np.zeros((n_norm, RES, RES), np.uint8), np.stack([
        (np.asarray(Image.open(m).convert("L").resize((RES, RES), Image.NEAREST)) > 127).astype(np.uint8)
        for m in mpaths])])

    out = {}
    for n in (4000, 23688):
        parts = {name: standardize(*load(run, n)) for name, run in
                 (("L8", "bigbank_L8"), ("L10_pool3", "bigbank_L10_pool3"))}
        fused_score = np.mean([p[0] for p in parts.values()], 0)
        fused_map = np.mean([p[1] for p in parts.values()], 0)
        out[f"proposed_n{n}"] = evaluate(fused_score, fused_map, y, n_norm, areas, masks)
        print(n, json.dumps({k: round(v, 4) for k, v in out[f"proposed_n{n}"].items()}), flush=True)

    # threshold of the paper's detector (100k random tokens, 3 seeds) and of the exhaustive L12 bank
    thr = []
    for s in (0, 1, 2):
        z = np.load(ROOT / f"outputs/tiled/raw/g1_518_s{s}.npz")
        thr.append(float(np.percentile(z["test_score"][:n_norm], 99)))
    out["baseline_100k_threshold"] = {"mean": float(np.mean(thr)), "std": float(np.std(thr))}
    (OUT / "metrics.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out["baseline_100k_threshold"]))


if __name__ == "__main__":
    main()
