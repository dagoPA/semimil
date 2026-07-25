#!/usr/bin/env python
"""Bootstrap 95% confidence intervals for image-level AUROC/AUPRC.

The MIIC test split is fixed and small (116 positives), so k-fold CV is neither
standard nor applicable (one-class training, all abnormals in test). Instead we
quantify test-set uncertainty with a stratified bootstrap (resampling positives
and negatives with replacement) over the saved per-image scores.

    uv run python scripts/bootstrap_ci.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path("/autofs/space/crater_001/projects/micropath/semi/outputs")
METHODS = {
    "A-DINOv2 (one-class)": ROOT / "patchcore/dinov2/image_scores.csv",
    "A-WRN50 (one-class)": ROOT / "patchcore/wrn50/image_scores.csv",
    "A-DINOv2-dominio": ROOT / "patchcore/dinov2_domain/image_scores.csv",
    "B-attn-MIL": ROOT / "mil/mil_dinov2/image_scores.csv",
    "PaDiM": ROOT / "padim/image_scores.csv",
    "DRAEM": ROOT / "draem/image_scores.csv",
    "STFPM": ROOT / "anomalib/stfpm/image_scores.csv",
    "FastFlow": ROOT / "anomalib/fastflow/image_scores.csv",
    "CFA": ROOT / "anomalib/cfa/image_scores.csv",
}
N_BOOT = 2000
SEED = 42


def load(path):
    y, s = [], []
    for r in csv.DictReader(path.open()):
        y.append(int(r["label"]))
        s.append(float(r["score"]))
    return np.array(y), np.array(s)


def boot_ci(y, s, rng):
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    au, ap = [], []
    for _ in range(N_BOOT):
        idx = np.concatenate([rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)])
        au.append(roc_auc_score(y[idx], s[idx]))
        ap.append(average_precision_score(y[idx], s[idx]))
    return np.array(au), np.array(ap)


def main() -> None:
    rng = np.random.default_rng(SEED)
    print(f"{'método':<24}{'AUROC [95% CI]':>26}{'AUPRC [95% CI]':>26}")
    out = []
    for name, path in METHODS.items():
        if not path.exists():
            continue
        y, s = load(path)
        au, ap = boot_ci(y, s, rng)
        a_lo, a_hi = np.percentile(au, [2.5, 97.5])
        p_lo, p_hi = np.percentile(ap, [2.5, 97.5])
        a0 = roc_auc_score(y, s)
        p0 = average_precision_score(y, s)
        print(
            f"{name:<24}{f'{a0:.3f} [{a_lo:.3f}, {a_hi:.3f}]':>26}"
            f"{f'{p0:.3f} [{p_lo:.3f}, {p_hi:.3f}]':>26}"
        )
        out.append({
            "method": name,
            "auroc": float(a0), "auroc_ci": [float(a_lo), float(a_hi)],
            "auprc": float(p0), "auprc_ci": [float(p_lo), float(p_hi)],
        })
    d = ROOT / "stats"
    d.mkdir(parents=True, exist_ok=True)
    (d / "bootstrap.json").write_text(json.dumps({"n_boot": N_BOOT, "results": out}, indent=2))
    print(f"\nsaved -> {d / 'bootstrap.json'}")


if __name__ == "__main__":
    main()
