#!/usr/bin/env python
"""Side-by-side qualitative comparison of the baseline detector and attention-MIL.

Columns: original | ground truth (red contour) | baseline detector map |
attention-MIL map. The rows are the largest real defects that attention-MIL detects,
so both methods are shown where both succeed. The one-class map is the stitched
token-distance map of the main-text baseline detector (first seed) and the MIL map is
the attention of the model whose scores Appendix B reports; nothing is retrained.

Reproducible & argument-free: reads ``configs/comparison.yaml``.

    uv run python scripts/plot_comparison.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from scipy.ndimage import gaussian_filter

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_patchcore import list_images, load_mask

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "comparison.yaml"


def norm(a):
    return (a - a.min()) / (np.ptp(a) + 1e-8)


def clip_norm(a, lo=60, hi=99.7):
    """Percentile-clipped normalization: suppress background, keep strong responses."""
    a = np.asarray(a, float)
    lov, hiv = np.percentile(a, lo), np.percentile(a, hi)
    return np.clip((a - lov) / (hiv - lov + 1e-8), 0, 1)


def upsample(grid_map, size, sigma):
    t = torch.tensor(np.asarray(grid_map), dtype=torch.float32)[None, None]
    up = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return gaussian_filter(up[0, 0].numpy(), sigma)


def main() -> None:
    cfg = yaml.safe_load(CONFIG_PATH.open())
    raw = Path(cfg["data"]["raw_dir"])
    d = cfg["data"]
    size, sigma = cfg["display_size"], cfg["sigma"]
    n_norm = len(list_images(sorted(Path(raw, d["test_normal_dir"]).iterdir())))
    abn = list_images(sorted(Path(raw, d["test_abnormal_dir"]).iterdir()))
    mask_dir = Path(raw, d["test_mask_dir"])

    # stored maps and scores; both files list the normal test images first, then the defective
    oc_maps = np.load(cfg["one_class_maps"])["test_map"][n_norm:]
    mil_dir = Path(cfg["mil_dir"])
    attn = np.load(mil_dir / "attention_test.npz")["attention"][n_norm:]
    with open(mil_dir / "image_scores.csv", newline="") as fh:
        scores = {r["name"]: float(r["score"]) for r in csv.DictReader(fh)}
    probs = np.array([scores[p.name] for p in abn])
    assert len(oc_maps) == len(attn) == len(abn)

    # largest defects the attention-MIL detects -> both localize cleanly
    areas = np.array([load_mask(mask_dir / f"{p.stem}_mask.jpg", 64).mean() for p in abn])
    detected = np.where(probs >= 0.5)[0]
    picks = list(detected[np.argsort(areas[detected])[::-1]][: cfg["n_rows"]])

    n = len(picks)
    fig, axes = plt.subplots(n, 4, figsize=(12, 3.05 * n))
    cols = ["original", "ground truth", "baseline detector", "attention-MIL"]
    for j, c in enumerate(cols):
        axes[0, j].set_title(c, fontsize=12)
    for r, k in enumerate(picks):
        pth = abn[k]
        img = np.asarray(Image.open(pth).convert("L").resize((size, size)))
        mask = load_mask(mask_dir / f"{pth.stem}_mask.jpg", size)
        a_map = clip_norm(upsample(oc_maps[k], size, sigma), lo=90)  # suppress background texture
        b_map = norm(upsample(attn[k], size, sigma))

        axes[r, 0].imshow(img, cmap="gray")
        axes[r, 1].imshow(img, cmap="gray")
        axes[r, 1].contour(mask, levels=[0.5], colors="red", linewidths=1.3)
        axes[r, 2].imshow(img, cmap="gray")
        axes[r, 2].imshow(a_map, cmap="jet", alpha=0.5)
        axes[r, 3].imshow(img, cmap="gray")
        axes[r, 3].imshow(b_map, cmap="jet", alpha=0.5)
        tag = "detects" if probs[k] >= 0.5 else "MISSES"
        axes[r, 0].set_ylabel(
            f"{pth.name}\ndefect {mask.mean() * 100:.2f}%\nMIL: {tag} (p={probs[k]:.2f})",
            fontsize=8,
        )
        for j in range(4):
            axes[r, j].set_xticks([])
            axes[r, j].set_yticks([])
    # no title inside the figure: the journal puts the title in the caption
    fig.tight_layout()
    out = Path(cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
