#!/usr/bin/env python
"""External-dataset examples: what Carinthia-S, Texture-AD Wafer and MVTec AD look like.

One row per dataset: two defect-free images and four defective ones with the reference
contour in green and the defect area as a percentage of the image. Each defective image
is the one with the median defect area among a fixed-seed sample of its class, so the
figure is deterministic. Carinthia-S images are shown after the border crop used in every
analysis. Reads the manifests written by scripts/prepare_external.py.

Argument-free: reads configs/external_examples.yaml.

    uv run python scripts/plot_external_examples.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "external_examples.yaml"

INK = "#1a1c20"
MUTED = "#566070"
CONTOUR = "#00dc00"


def load(path: str, crop: int, mode: str) -> np.ndarray:
    im = Image.open(path).convert(mode)
    if crop:
        im = im.crop((crop, crop, im.width - crop, im.height - crop))
    return np.asarray(im)


def pick(rows, cls, label, rng, n_cand, crop):
    """A defect-free image, or the defective image with the median defect area."""
    pool = [r for r in rows if r["cls"] == cls and r["label"] == label and (label == "0" or r["mask"])]
    pool.sort(key=lambda r: r["image"])
    if label == "0":
        return pool[int(rng.integers(len(pool)))], None
    cand = [pool[i] for i in rng.choice(len(pool), min(n_cand, len(pool)), replace=False)]
    areas = [(load(r["mask"], crop, "L") > 127).mean() for r in cand]
    i = int(np.argsort(areas)[len(areas) // 2])
    return cand[i], areas[i]


def main() -> None:
    cfg = yaml.safe_load(CONFIG_PATH.open())
    rng = np.random.default_rng(cfg["seed"])
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7})

    W, left, right, top, gap, lab, rgap = 7.0, 0.36, 0.04, 0.24, 0.05, 0.17, 0.09
    n = 6
    t = (W - left - right - (n - 1) * gap) / n
    rows_n = len(cfg["datasets"])
    H = top + rows_n * (t + lab) + (rows_n - 1) * rgap + 0.02
    fig = plt.figure(figsize=(W, H))

    def axes(r, c):
        x = left + c * (t + gap)
        y = H - top - r * (t + lab + rgap) - t
        ax = fig.add_axes((x / W, y / H, t / W, t / H))
        ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.4), s.set_color(MUTED)
        return ax, x, y

    for r, d in enumerate(cfg["datasets"]):
        rows = list(csv.DictReader(open(Path(cfg["manifests"]) / d["manifest"])))
        picks = [pick(rows, c, "0", rng, cfg["candidates"], d["crop"]) for c in d["normal"]]
        picks += [pick(rows, c, "1", rng, cfg["candidates"], d["crop"]) for c in d["defective"]]
        for c, ((row, area), cls) in enumerate(zip(picks, d["normal"] + d["defective"])):
            ax, x, y = axes(r, c)
            ax.imshow(load(row["image"], d["crop"], "RGB"))
            name = d["label"].format(cls=cls).replace("_", " ")
            if area is not None:
                m = load(row["mask"], d["crop"], "L") > 127
                ax.contour(m, levels=[0.5], colors=CONTOUR, linewidths=0.8)
                name = f"{name}, {100 * area:.2g} %"
            fig.text((x + t / 2) / W, (y - 0.035) / H, name, ha="center", va="top", color=INK)
        y_mid = H - top - r * (t + lab + rgap) - t / 2
        fig.text(0.13 / W, y_mid / H, d["name"], rotation=90, ha="center", va="center",
                 fontsize=7.5, fontweight="bold", color=INK)
        fig.text(0.27 / W, y_mid / H, d["subtitle"], rotation=90, ha="center", va="center",
                 fontsize=6.5, color=MUTED)

    # column groups, and a rule between the defect-free and the defective images
    x_sep = left + 2 * t + 1.5 * gap
    fig.text((left + t + gap / 2) / W, (H - 0.12) / H, "Defect-free", ha="center", va="center",
             fontsize=7.5, fontweight="bold", color=INK)
    fig.text((x_sep + (W - right - x_sep) / 2) / W, (H - 0.12) / H,
             "Defective (reference contour in green)", ha="center", va="center", fontsize=7.5,
             fontweight="bold", color=INK)
    fig.add_artist(plt.Line2D([x_sep / W] * 2, [0.01, (H - top + 0.02) / H], color=MUTED, lw=0.6))

    out = Path(cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=500)  # about 650 dpi at the text width of the preprint
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
