#!/usr/bin/env python
"""Dataset-overview figure: what MIIC looks like, and how small the defects get.

Top row: defect-free images, showing the strong periodic IC structure a detector
must not mistake for an anomaly. Bottom row: defective images with the
ground-truth contour in red, picked to span the size range that makes this
dataset hard -- the largest defect, a mid-sized one, and the smallest.

Selection is by defect area, so the figure is deterministic and matches the
sizes quoted in the caption.

    uv run python scripts/plot_dataset_overview.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "patchcore.yaml"
OUT = Path("/autofs/space/crater_001/projects/micropath/semi/outputs/report/assets/dataset_overview.png")

INK = "#1a1c20"
MUTED = "#566070"
N_COL = 3


def main() -> None:
    cfg = yaml.safe_load(CONFIG_PATH.open())
    raw = Path(cfg["data"]["raw_dir"])
    d = cfg["data"]
    norm_dir = raw / d["test_normal_dir"]
    abn_dir = raw / d["test_abnormal_dir"]
    mask_dir = raw / d["test_mask_dir"]

    exts = {".jpg", ".jpeg", ".png"}
    normals = sorted(p for p in norm_dir.iterdir() if p.suffix.lower() in exts)
    abnormals = sorted(p for p in abn_dir.iterdir() if p.suffix.lower() in exts)
    if not normals or not abnormals:
        sys.exit(f"no images under {norm_dir} / {abn_dir}")

    # defect area per abnormal image, to pick the largest / median / smallest
    areas = []
    for p in abnormals:
        mp = mask_dir / f"{p.stem}_mask.jpg"
        if not mp.exists():
            continue
        m = np.asarray(Image.open(mp).convert("L")) > 127
        areas.append((m.mean(), p, mp))
    areas.sort(key=lambda r: -r[0])
    picks = [areas[0], areas[len(areas) // 2], areas[-1]]

    fig, axes = plt.subplots(2, N_COL, figsize=(9.6, 6.9))
    fig.suptitle("MIIC — integrated-circuit microscopy images (512×512, grayscale)",
                 fontsize=12, color=INK, y=0.975)

    for j in range(N_COL):
        ax = axes[0, j]
        ax.imshow(Image.open(normals[j]).convert("L"), cmap="gray")
        ax.set_title("defect-free", fontsize=10, color=INK, pad=6)
        ax.set_xticks([]); ax.set_yticks([])

    for j, (frac, ip, mp) in enumerate(picks):
        ax = axes[1, j]
        ax.imshow(Image.open(ip).convert("L"), cmap="gray")
        mask = np.asarray(Image.open(mp).convert("L")) > 127
        ax.contour(mask, levels=[0.5], colors="red", linewidths=1.4)
        ax.set_xlabel(f"defective · defect {frac * 100:.2f} % of area",
                      fontsize=9.5, color=INK, labelpad=6)
        ax.set_xticks([]); ax.set_yticks([])

    axes[0, 0].set_ylabel("DEFECT-FREE", fontsize=10, color=MUTED, labelpad=8)
    axes[1, 0].set_ylabel("DEFECTIVE (contour = GT)", fontsize=10, color=MUTED, labelpad=8)

    fig.tight_layout(rect=(0, 0, 1, 0.955))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"saved -> {OUT}  (defect sizes: "
          f"{', '.join(f'{f * 100:.2f}%' for f, _, _ in picks)})")


if __name__ == "__main__":
    main()
