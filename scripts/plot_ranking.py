#!/usr/bin/env python
"""Forest plot of the method ranking with 95% bootstrap CIs (image AUROC).

Reads the saved bootstrap + DeLong results and renders a ranked point-and-error-
bar figure. Identity is encoded by color and marker shape (blue circle = proposed
detector, gray circles = the baseline detector with three feature extractors, open
squares = other methods); a band marks the CI of the proposed detector, and "n.s."
marks a method whose AUROC does not differ from it after Holm correction.

    uv run python scripts/plot_ranking.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Appendix B statistics under the protocol of the main text (scripts/analysis/appendix_b_stats.py)
STATS = Path("/autofs/space/crater_001/projects/micropath/semi/outputs/appendix_b/stats")
OUT = Path("/autofs/space/crater_001/projects/micropath/semi/outputs/report/assets/ranking.png")

BLUE = "#2b6cb0"     # proposed detector (as in the headline figure)
LIGHT = "#8d97a8"    # baseline detector with its three feature extractors
GRAY = "#5f6878"     # other methods
INK = "#1a1c20"
MUTED = "#566070"
GRID = "#e4e7ec"
REF = "Proposed"

BASELINES = {"Baseline (DINOv2)", "Baseline (WRN50)", "Baseline (adapted DINOv2)"}
LABEL = {
    "Proposed": "Proposed",
    "Baseline (DINOv2)": "Baseline detector (DINOv2)",
    "Baseline (WRN50)": "Baseline detector (WRN50)",
    "Baseline (adapted DINOv2)": "Baseline detector (adapted)",
    "Attention-MIL": "Attention-MIL",
    "PatchCore": "PatchCore ‡",
    "FastFlow": "FastFlow ‡",
    "PaDiM": "PaDiM",
    "STFPM": "STFPM ‡",
    "CFA": "CFA ‡",
    "DRAEM": "DRAEM",
}


def main() -> None:
    boot = {r["method"]: r for r in json.loads((STATS / "bootstrap.json").read_text())["results"]}
    delong = json.loads((STATS / "delong.json").read_text())
    pmap = {c["method"]: c["p_holm"] for c in delong["comparisons"]}

    rows = sorted(boot.values(), key=lambda r: r["auroc"], reverse=True)
    ref = boot[REF]

    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    ax.axvspan(ref["auroc_ci"][0], ref["auroc_ci"][1], color=BLUE, alpha=0.07, zorder=0)
    ax.axvline(ref["auroc"], color=BLUE, lw=1, ls="--", alpha=0.5, zorder=1)

    n = len(rows)
    for i, r in enumerate(rows):
        y = n - 1 - i
        m = r["method"]
        color = BLUE if m == REF else (LIGHT if m in BASELINES else GRAY)
        lo, hi = r["auroc_ci"]
        ax.plot([lo, hi], [y, y], color=color, lw=2, zorder=2, solid_capstyle="round")
        if m == REF or m in BASELINES:
            ax.plot(r["auroc"], y, "o", color=color, ms=10 if m == REF else 8,
                    mec="white", mew=1.2, zorder=3)
        else:
            ax.plot(r["auroc"], y, "s", color="white", ms=8, mec=color, mew=2, zorder=3)
        ax.text(hi + 0.006, y, f"{r['auroc']:.3f}", va="center", ha="left",
                fontsize=8.5, color=INK, fontweight="bold" if m == REF else "normal")
        # significance against the proposed detector (DeLong, Holm-corrected)
        p = pmap.get(m)
        tag = "ref" if m == REF else ("n.s." if (p is not None and p >= 0.05) else "")
        if tag:
            ax.text(lo - 0.006, y, tag, va="center", ha="right", fontsize=7.5,
                    color=MUTED, style="italic")

    ax.set_yticks(range(n))
    ax.set_yticklabels([LABEL[r["method"]] for r in rows][::-1], fontsize=9)
    for t, r in zip(ax.get_yticklabels(), rows[::-1], strict=True):
        if r["method"] == REF:
            t.set_color(BLUE), t.set_fontweight("bold")
    ax.set_xlabel("Image AUROC  (95 % bootstrap CI, 2000 resamples)", fontsize=9.5)
    ax.set_xlim(0.50, 1.005)
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(length=0)

    # legend (shape + color), redundant with the labels
    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], marker="o", color=BLUE, lw=0, mfc=BLUE, mec="white", ms=9,
               label="Proposed detector"),
        Line2D([0], [0], marker="o", color=LIGHT, lw=0, mfc=LIGHT, mec="white", ms=8,
               label="Baseline detector, three feature extractors"),
        Line2D([0], [0], marker="s", color=GRAY, lw=0, mfc="white", mec=GRAY, mew=2, ms=8,
               label="Other methods  (‡ reproduced with anomalib)"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=8.5)
    # no title inside the figure: the journal puts the title in the caption
    fig.tight_layout()
    fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")  # vector, journal artwork
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
