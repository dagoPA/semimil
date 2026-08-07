#!/usr/bin/env python
"""Per-epoch image AUROC of the reproduced baselines (small multiples).

Shows why the literature numbers on MIIC are optimistic: validated on the full
test each epoch, the AUROC swings widely; the BEST epoch (selected using the test,
as ref [14] does) reaches the reported value, while any fixed/last checkpoint sits
well below. Our one-class DINOv2 detector (red line) matches or exceeds every peak
WITHOUT ever using the test for model selection.

    uv run python scripts/plot_epochsel.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path("/autofs/space/crater_001/projects/micropath/semi/outputs")
ES = ROOT / "anomalib_epochsel"
OUT = ROOT / "report/assets/epochsel.png"

RED = "#cf4128"      # our method
STEEL = "#2f6a9c"    # baseline curve
INK = "#1a1c20"
MUTED = "#566070"
GRID = "#e4e7ec"

ORDER = ["stfpm", "fastflow", "cfa"]
TITLE = {"stfpm": "STFPM", "fastflow": "FastFlow", "cfa": "CFA"}


def main() -> None:
    summary = json.loads((ES / "summary.json").read_text())
    boot = {r["method"]: r for r in
            json.loads((ROOT / "stats/bootstrap.json").read_text())["results"]}
    ref = boot["DINOv2 (one-class)"]["auroc"]

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.7), sharey=True)

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

        # reference lines: literature (neutral dashed) and our method (red solid)
        ax.axhline(lit, color=MUTED, lw=1.2, ls=(0, (5, 3)), zorder=2)
        ax.axhline(ref, color=RED, lw=1.7, zorder=2)

        # per-epoch curve
        ax.plot(xs, ys, color=STEEL, lw=1.7, marker="o", ms=3, mfc=STEEL,
                mec="white", mew=0.4, zorder=3, solid_capstyle="round")

        # best-epoch (test-selected) and last-epoch (leakage-free) markers
        ax.plot(best_x, best_y, "o", ms=11, mfc=STEEL, mec="white", mew=1.6, zorder=5)
        ax.plot(last_x, last_y, "o", ms=9, mfc="white", mec=STEEL, mew=2, zorder=5)

        ax.annotate(f"best epoch: {best_y:.3f}", (best_x, best_y),
                    textcoords="offset points", xytext=(0, 12), ha="center",
                    fontsize=8, color=INK, fontweight="bold")
        ax.annotate(f"last: {last_y:.3f}", (last_x, last_y),
                    textcoords="offset points", xytext=(-4, -14), ha="right",
                    fontsize=8, color=MUTED)
        # label the literature line at the left edge (best/last markers sit
        # on the curve, right/interior, so this never collides with them)
        ax.text(1.0, lit - 0.006, f"reported [14]: {lit:.3f}",
                ha="left", va="top", fontsize=7.5, color=MUTED, style="italic")

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
    axes[0].set_ylim(0.70, 1.005)

    handles = [
        Line2D([0], [0], color=STEEL, lw=1.7, marker="o", ms=4, mfc=STEEL,
               mec="white", label="AUROC per epoch (baseline)"),
        Line2D([0], [0], color=MUTED, lw=1.2, ls=(0, (5, 3)),
               label="reported by [14]"),
        Line2D([0], [0], color=RED, lw=1.7,
               label=f"DINOv2 one-class (ours), no selection on test ({ref:.3f})"),
        Line2D([0], [0], color=STEEL, lw=0, marker="o", ms=9, mfc=STEEL,
               mec="white", mew=1.4, label="best epoch (selected on the test set)"),
        Line2D([0], [0], color=STEEL, lw=0, marker="o", ms=8, mfc="white",
               mec=STEEL, mew=2, label="last epoch (leakage-free)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Reported AUROC depends on selecting the epoch on the test set",
                 fontsize=11.5, color=INK, y=1.0)
    fig.tight_layout(rect=(0, 0.02, 1, 0.98))
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
