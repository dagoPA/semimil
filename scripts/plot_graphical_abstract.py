#!/usr/bin/env python
"""Graphical abstract for IEEE Latin America Transactions (illustrated pipeline).

Wide banner (>=1328x531 px, >=400 dpi) that VISUALIZES the one-class MIL method:
image -> patch grid (bag) -> frozen DINOv2 -> patch feature vectors -> memory bank
with nearest-neighbor distance -> anomaly score/map -> headline results.

    uv run --with pillow python scripts/plot_graphical_abstract.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle
from PIL import Image
from scipy.ndimage import gaussian_filter

ROOT = Path("/autofs/space/crater_001/projects/micropath/semi")
IMG = ROOT / "MIIC/Anomaly_test/abnormal_img/test_abnormal_00053.jpg"
MASK = ROOT / "MIIC/Anomaly_test/abnormal_mask/test_abnormal_00053_mask.jpg"
OUT = ROOT / "outputs/report/assets/graphical_abstract.png"

RED = "#cf4128"
STEEL = "#2f6a9c"
INK = "#1a1c20"
MUTED = "#566070"

DPI = 400
W_IN, H_IN = 2656 / DPI, 1062 / DPI

rng = np.random.default_rng(7)


def arrow(ax, x0, x1, y, color=MUTED, lw=1.7):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=12,
                                 lw=lw, color=color, shrinkA=0, shrinkB=0))


def feat_vector(ax, x, y0, w, cellh, vals, *, highlight=False):
    """Draw a feature embedding as a vertical column of color-mapped cells."""
    cmap = plt.get_cmap("viridis")
    n = len(vals)
    for i, v in enumerate(vals):
        ax.add_patch(Rectangle((x, y0 + i * cellh), w, cellh, fc=cmap(v),
                               ec="white", lw=0.4))
    ec = RED if highlight else "#9aa2ad"
    ax.add_patch(Rectangle((x, y0), w, n * cellh, fc="none", ec=ec,
                           lw=1.8 if highlight else 0.8))


def main() -> None:
    im = np.asarray(Image.open(IMG).convert("L"))
    mask = np.asarray(Image.open(MASK).convert("L")) > 127
    H = im.shape[0]
    ys, xs = np.where(mask)
    cyf, cxf = ys.mean() / H, xs.mean() / H  # defect centroid (fractions)

    fig = plt.figure(figsize=(W_IN, H_IN), dpi=DPI)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # Matches the paper title; the flow below already says what the method is.
    ax.text(0.5, 0.945, "The Feature Extractor Decides", ha="center", va="center",
            fontsize=13.0, color=INK, fontweight="bold")

    yc = 0.50  # vertical centre of the flow

    # ---- Stage 1: image with patch grid, defect patch highlighted ----
    axi = fig.add_axes([0.015, 0.24, 0.165, 0.52])
    axi.imshow(im, cmap="gray"); axi.set_xticks([]); axi.set_yticks([])
    G = 6
    for k in range(1, G):
        axi.axhline(k * H / G, color="white", lw=0.6, alpha=0.7)
        axi.axvline(k * H / G, color="white", lw=0.6, alpha=0.7)
    gi, gj = int(cyf * G), int(cxf * G)
    axi.add_patch(Rectangle((gj * H / G, gi * H / G), H / G, H / G, fc="none",
                            ec=RED, lw=2.2))
    axi.contour(mask, levels=[0.5], colors=RED, linewidths=0.9, alpha=0.85)
    for s in axi.spines.values():
        s.set_edgecolor("#c9cdd4")
    ax.text(0.0975, 0.155, "bag of patches (MIL)", ha="center", fontsize=6.6,
            color=INK, fontweight="bold")
    ax.text(0.0975, 0.105, "one patch is anomalous", ha="center",
            fontsize=6.0, color=MUTED)

    arrow(ax, 0.19, 0.235, yc)

    # ---- Stage 2: frozen DINOv2 -> patch feature vectors ----
    ax.text(0.315, 0.735, "frozen DINOv2", ha="center", fontsize=7.6, color=STEEL,
            fontweight="bold")
    ax.text(0.315, 0.685, "(no training)", ha="center", fontsize=6.6, color=MUTED)
    cellh, nrow, vw = 0.052, 6, 0.026
    y0 = yc - nrow * cellh / 2
    xsv = [0.258, 0.298, 0.338, 0.386]
    for k, xv in enumerate(xsv):
        hl = (k == 3)
        feat_vector(ax, xv, y0, vw, cellh, rng.random(nrow), highlight=hl)
        if hl:
            ax.text(xv + vw / 2, y0 - 0.055, "defect", ha="center", fontsize=6.0,
                    color=RED, fontweight="bold")
    ax.text(0.335, 0.145, "patch feature vectors", ha="center", fontsize=7.2,
            color=INK, fontweight="bold")

    arrow(ax, 0.43, 0.475, yc)

    # ---- Stage 3: memory bank + nearest-neighbor distance ----
    axm = fig.add_axes([0.49, 0.24, 0.155, 0.5])
    normal = rng.normal(0, 1, (60, 2)) @ [[1.0, 0.3], [0.3, 0.7]]
    axm.scatter(normal[:, 0], normal[:, 1], s=10, c=STEEL, alpha=0.55,
                edgecolors="white", linewidths=0.3)
    test = np.array([2.7, 2.2])
    d = np.linalg.norm(normal - test, axis=1); nn = normal[d.argmin()]
    axm.plot([test[0], nn[0]], [test[1], nn[1]], "--", color=RED, lw=1.3)
    axm.scatter(*test, s=52, c=RED, edgecolors="white", linewidths=0.8, zorder=5)
    axm.set_xticks([]); axm.set_yticks([]); axm.axis("off")
    axm.set_title("memory bank\nof normal patches", fontsize=6.6, color=MUTED, pad=1)
    ax.text(0.5675, 0.155, "distance to nearest", ha="center", fontsize=7.2,
            color=INK, fontweight="bold")
    ax.text(0.5675, 0.108, "normal = anomaly score", ha="center", fontsize=6.6,
            color=MUTED)

    arrow(ax, 0.655, 0.70, yc)

    # ---- Stage 4: anomaly map on the image ----
    axh = fig.add_axes([0.705, 0.26, 0.135, 0.48])
    heat = np.zeros_like(im, dtype=float); heat[mask] = 1.0
    heat = gaussian_filter(heat, 14)
    axh.imshow(im, cmap="gray")
    axh.imshow(np.ma.masked_less(heat / heat.max(), 0.12), cmap="jet", alpha=0.6)
    axh.set_xticks([]); axh.set_yticks([])
    for s in axh.spines.values():
        s.set_edgecolor("#c9cdd4")
    ax.text(0.7725, 0.175, "anomaly map", ha="center", fontsize=7.2, color=INK,
            fontweight="bold")

    ax.text(0.5, 0.045, "no labels   ·   no training   ·   frozen backbone",
            ha="center", fontsize=7.0, color=RED, style="italic")

    arrow(ax, 0.85, 0.872, yc)

    # ---- Stage 5: results ----
    # rx is bounded by the widest item in this column ("beats 8 competing",
    # 0.134 wide): centring it any further right pushes ink past the canvas.
    rx = 0.922
    ax.text(rx, 0.66, "image AUROC", ha="center", fontsize=6.8, color=MUTED)
    ax.text(rx, 0.545, "0.972", ha="center", fontsize=20, color=RED, fontweight="bold")
    # two lines: the full phrase would run past the right edge at this x
    ax.text(rx, 0.432, "beats 8 competing", ha="center", fontsize=6.8, color=INK)
    ax.text(rx, 0.378, "methods", ha="center", fontsize=6.8, color=INK)
    ax.text(rx, 0.320, "(DeLong · Holm)", ha="center", fontsize=5.8, color=MUTED)
    ax.text(rx, 0.235, "generalizes to", ha="center", fontsize=6.0, color=MUTED)
    ax.text(rx, 0.175, "MVTec AD 0.974", ha="center", fontsize=7.0, color=STEEL,
            fontweight="bold")

    fig.savefig(OUT, dpi=DPI, facecolor="white")
    sz = Image.open(OUT).size
    print(f"saved -> {OUT}  {sz[0]}x{sz[1]} px  ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
