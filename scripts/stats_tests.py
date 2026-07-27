#!/usr/bin/env python
"""Statistical comparison of methods via DeLong's test for correlated ROC AUCs.

DeLong's test (Sun & Xu, 2014, fast algorithm) compares two AUROCs computed on
the *same* test samples and returns a p-value for their difference. We compare
the reference method (one-class detector, DINOv2) against every other method
saved per-image scores.

    uv run python scripts/stats_tests.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

ROOT = Path("/autofs/space/crater_001/projects/micropath/semi/outputs")
METHODS = {
    "DINOv2 (one-class)": ROOT / "patchcore/dinov2/image_scores.csv",
    "WRN50 (one-class)": ROOT / "patchcore/wrn50/image_scores.csv",
    "DINOv2-dom (one-class)": ROOT / "patchcore/dinov2_domain/image_scores.csv",
    "Attention-MIL": ROOT / "mil/mil_dinov2/image_scores.csv",
    "PatchCore": ROOT / "anomalib/patchcore/image_scores.csv",
    "PaDiM": ROOT / "padim/image_scores.csv",
    "DRAEM": ROOT / "draem/image_scores.csv",
    "STFPM": ROOT / "anomalib/stfpm/image_scores.csv",
    "FastFlow": ROOT / "anomalib/fastflow/image_scores.csv",
    "CFA": ROOT / "anomalib/cfa/image_scores.csv",
}
REF = "DINOv2 (one-class)"


# ---- fast DeLong ----
def _midrank(x):
    j = np.argsort(x)
    z = x[j]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        k = i
        while k < n and z[k] == z[i]:
            k += 1
        t[i:k] = 0.5 * (i + k - 1) + 1
        i = k
    out = np.empty(n)
    out[j] = t
    return out


def _fast_delong(preds, n_pos):
    m = n_pos
    n = preds.shape[1] - m
    pos, neg = preds[:, :m], preds[:, m:]
    k = preds.shape[0]
    tx = np.array([_midrank(pos[r]) for r in range(k)])
    ty = np.array([_midrank(neg[r]) for r in range(k)])
    tz = np.array([_midrank(preds[r]) for r in range(k)])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    return aucs, np.atleast_2d(cov)


def delong_test(y, s1, s2):
    """Two-sided DeLong p-value for AUROC(s1) vs AUROC(s2) on labels y."""
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    order = np.concatenate([pos, neg])
    preds = np.vstack([s1[order], s2[order]])
    aucs, cov = _fast_delong(preds, len(pos))
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = (aucs[0] - aucs[1]) / np.sqrt(var) if var > 0 else 0.0
    p = 2 * stats.norm.sf(abs(z))
    return float(aucs[0]), float(aucs[1]), float(z), float(p)


def load(path):
    d = {}
    y = {}
    for r in csv.DictReader(path.open()):
        d[r["name"]] = float(r["score"])
        y[r["name"]] = int(r["label"])
    return d, y


def stars(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else "n.s."


def main() -> None:
    avail = {k: v for k, v in METHODS.items() if v.exists()}
    missing = [k for k in METHODS if k not in avail]
    if missing:
        print(f"(faltan, se omiten): {missing}\n")

    scores, labels = {}, {}
    for k, v in avail.items():
        scores[k], labels[k] = load(v)
    # common test images across all methods
    common = set.intersection(*[set(s) for s in scores.values()])
    names = sorted(common)
    y = np.array([labels[REF][n] for n in names])
    S = {k: np.array([scores[k][n] for n in names]) for k in avail}
    print(f"imágenes comunes: {len(names)}  ({int(y.sum())} con defecto)\n")

    print(f"{'método':<24}{'AUROC':>8}{'ΔvsRef':>9}{'p (DeLong)':>13}{'sig':>6}")
    rows = []
    ref_auc = roc_auc_score(y, S[REF])
    print(f"{REF:<24}{ref_auc:>8.3f}{'—':>9}{'(ref)':>13}{'':>6}")
    for k in avail:
        if k == REF:
            continue
        auc = roc_auc_score(y, S[k])
        _, _, z, p = delong_test(y, S[REF], S[k])
        delta = ref_auc - auc
        print(f"{k:<24}{auc:>8.3f}{delta:>+9.3f}{p:>13.2e}{stars(p):>6}")
        rows.append({"method": k, "auroc": auc, "delta_vs_ref": delta, "p_delong": p, "z": z})

    out = ROOT / "stats" / "delong.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"reference": REF, "ref_auroc": ref_auc, "comparisons": rows}
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
