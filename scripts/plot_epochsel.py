#!/usr/bin/env python
"""Per-epoch image AUROC of the reproduced baselines (small multiples).

Shows why the literature numbers on MIIC are optimistic: validated on the full
test each epoch, the AUROC swings widely; the BEST epoch (selected using the test,
as Barusco et al. do) reaches the reported value, while any fixed/last checkpoint sits
well below. The baseline detector (red line), which has no checkpoint to select,
matches or exceeds every peak without using the test set.

    uv run python scripts/plot_epochsel.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path("/autofs/space/crater_001/projects/micropath/semi/outputs")
ES = ROOT / "anomalib_epochsel"
OUT = ROOT / "report/assets/epochsel.png"

RED = "#cf4128"      # baseline detector
STEEL = "#2f6a9c"    # baseline curve
INK = "#1a1c20"
MUTED = "#566070"
GRID = "#e4e7ec"

ORDER = ["stfpm", "fastflow", "cfa"]
TITLE = {"stfpm": "STFPM", "fastflow": "FastFlow", "cfa": "CFA"}


def place_clear(fig, ax, label, xy, xs, ys):
    """Annotate xy at the first offset whose text box stays off the per-epoch curve."""
    fine = np.linspace(0, len(xs) - 1, 40 * len(xs))
    curve = np.c_[np.interp(fine, range(len(xs)), xs), np.interp(fine, range(len(xs)), ys)]
    for dx, dy in ((-4, -14), (-4, 10), (-4, -30), (-4, -46), (-4, -62)):
        t = ax.annotate(label, xy, textcoords="offset points", xytext=(dx, dy), ha="right",
                        fontsize=8, color=MUTED)
        fig.canvas.draw()
        box = t.get_window_extent().padded(5 * fig.dpi / 72)  # 5 pt around the text
        px, py = ax.transData.transform(curve).T
        if not ((px >= box.x0) & (px <= box.x1) & (py >= box.y0) & (py <= box.y1)).any():
            return
        t.remove()
    ax.annotate(label, xy, textcoords="offset points", xytext=(-4, -14), ha="right",
                fontsize=8, color=MUTED)


def main() -> None:
    summary = json.loads((ES / "summary.json").read_text())
    boot = {r["method"]: r for r in
            json.loads((ROOT / "appendix_b/stats/bootstrap.json").read_text())["results"]}
    ref = boot["Baseline (DINOv2)"]["auroc"]

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.7), sharey=True)
    axes[0].set_ylim(0.70, 1.005)
    last_labels = []

    for ax, name in zip(axes, ORDER, strict=True):
        curve = json.loads((ES / name / "curve.json").read_text())
        curve = [c for c in curve if c.get("auroc") is not None]
        xs = [c["epoch"] + 1 for c in curve]
        ys = [c["auroc"] for c in curve]
        s = summary[name]
        lit = s["literature_auroc"]
        best_x = s["best_epoch"] + 1
        best_y = s["best_epoch_auroc"]
        last_x, last_y = xs[-1], ys[-1]

        # reference lines: literature (neutral dashed) and baseline detector (red solid)
        ax.axhline(lit, color=MUTED, lw=1.2, ls=(0, (5, 3)), zorder=2)
        ax.axhline(ref, color=RED, lw=1.7, zorder=2)

        # per-epoch curve
        ax.plot(xs, ys, color=STEEL, lw=1.7, marker="o", ms=3, mfc=STEEL,
                mec="white", mew=0.4, zorder=3, solid_capstyle="round")

        # best-epoch (test-selected) and last-epoch (leakage-free) markers
        ax.plot(best_x, best_y, "o", ms=11, mfc=STEEL, mec="white", mew=1.6, zorder=5)
        ax.plot(last_x, last_y, "o", ms=9, mfc="white", mec=STEEL, mew=2, zorder=5)

        # a best epoch near the left edge gets a left-aligned label that stays in the panel
        early = best_x < 0.15 * len(xs)
        ax.annotate(f"best epoch: {best_y:.3f}", (best_x, best_y),
                    textcoords="offset points", xytext=(-6 if early else 0, 12),
                    ha="left" if early else "center", fontsize=8, color=INK, fontweight="bold")
        last_labels.append((ax, f"last: {last_y:.3f}", (last_x, last_y), xs, ys))
        # label the literature line on the side away from the red line: below it at the
        # left edge when it lies under the red line, above it at the right edge otherwise
        if lit < ref:
            ax.text(1.0, lit - 0.006, f"reported: {lit:.3f}",
                    ha="left", va="top", fontsize=7.5, color=MUTED, style="italic")
        else:
            ax.text(len(xs), lit + 0.004, f"reported: {lit:.3f}",
                    ha="right", va="bottom", fontsize=7.5, color=MUTED, style="italic")

        ax.set_title(TITLE[name], fontsize=11, color=INK, pad=8)
        ax.set_xlabel("epoch", fontsize=9)
        ax.set_xlim(0.5, len(xs) + 0.5)
        ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(MUTED)
        ax.tick_params(length=0, labelsize=8)

    axes[0].set_ylabel("Image AUROC (test)", fontsize=9.5)

    handles = [
        Line2D([0], [0], color=STEEL, lw=1.7, marker="o", ms=4, mfc=STEEL,
               mec="white", label="AUROC per epoch (reproduced method)"),
        Line2D([0], [0], color=MUTED, lw=1.2, ls=(0, (5, 3)),
               label="reported by Barusco et al."),
        Line2D([0], [0], color=RED, lw=1.7,
               label=f"baseline detector, no selection on the test set ({ref:.3f})"),
        Line2D([0], [0], color=STEEL, lw=0, marker="o", ms=9, mfc=STEEL,
               mec="white", mew=1.4, label="best epoch (selected on the test set)"),
        Line2D([0], [0], color=STEEL, lw=0, marker="o", ms=8, mfc="white",
               mec=STEEL, mew=2, label="last epoch (leakage-free)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, -0.06))
    # no title inside the figure: the journal puts the title in the caption
    fig.tight_layout(rect=(0, 0.02, 1, 0.98))
    for args in last_labels:  # placed on the final layout, clear of the curves
        place_clear(fig, *args)
    fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")  # vector, journal artwork
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
