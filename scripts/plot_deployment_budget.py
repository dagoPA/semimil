#!/usr/bin/env python
"""Deployment-budget figure: how few normal images, and how dirty a pool, suffice.

Three panels from scripts/deployment_budget.py:
  (a) image AUROC vs the number of normal images in the memory bank,
  (b) AUROC vs pool contamination with synthetic anomalies,
  (c) the same with real anomalies held out of the test set.

Identity is encoded by BOTH color and marker shape (filled circle = raw bank,
open square = self-cleaned), so the panels never rely on color alone. The
palette is the one used by the other figures, checked for CVD separation.

    uv run python scripts/plot_deployment_budget.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/autofs/space/crater_001/projects/micropath/semi/outputs")
RES = ROOT / "deployment_budget/results.json"
OUT = ROOT / "report/assets/budget.png"

RED = "#cf4128"      # raw bank
STEEL = "#2f6a9c"    # self-cleaned
INK = "#1a1c20"
MUTED = "#566070"
GRID = "#e4e7ec"

BASE_RATE = 0.46     # MIIC's own anomaly rate (%), for reference
# best reproduced external baseline under the clean protocol (Table VIII), which
# trains on all 23,888 normals -- the bar the few-shot bank has to clear
BEST_BASELINE = 0.908
BEST_BASELINE_NAME = "FastFlow"


def agg(rows, key="image_auroc"):
    """mean and std over seeds."""
    a = np.array([r[key] for r in rows])
    return a.mean(), (a.std(ddof=1) if len(a) > 1 else 0.0), len(a)


def style(ax):
    ax.set_facecolor("white")
    ax.grid(True, color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=3)


def main() -> None:
    res = json.loads(RES.read_text())
    OUT.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.45))

    # ---------------- (a) normal-image budget -------------------------------
    ax = axes[0]
    style(ax)
    bud = [r for r in res if r["part"] == "budget"]
    ks = sorted({r["k"] for r in bud})
    mu = np.array([agg([r for r in bud if r["k"] == k])[0] for k in ks])
    sd = np.array([agg([r for r in bud if r["k"] == k])[1] for k in ks])
    full = mu[-1]

    ax.fill_between(ks, mu - sd, mu + sd, color=RED, alpha=0.16, lw=0, zorder=2)
    ax.plot(ks, mu, color=RED, lw=2.0, marker="o", ms=5.5, mec="white", mew=1.2, zorder=4)
    ax.axhline(full, color=MUTED, lw=1.0, ls="--", alpha=0.7, zorder=1)
    ax.text(ks[-1], full + 0.006, f"full bank (k={ks[-1]}): {full:.3f}",
            color=MUTED, fontsize=7.6, va="bottom", ha="right")

    # the operationally meaningful point: the smallest bank that already beats the
    # best baseline trained on all 23,888 normals
    ax.axhline(BEST_BASELINE, color=STEEL, lw=1.0, ls=":", alpha=0.9, zorder=1)
    ax.text(ks[-1], BEST_BASELINE + 0.005, f"{BEST_BASELINE_NAME}: {BEST_BASELINE:.3f}",
            color=STEEL, fontsize=7.6, va="bottom", ha="right")
    cross = next((k for k, m in zip(ks, mu, strict=True) if m > BEST_BASELINE), None)
    if cross is not None:
        mc = mu[ks.index(cross)]
        ax.plot([cross], [mc], marker="o", ms=12, mfc="none", mec=RED, mew=1.8, zorder=5)
        ax.annotate(f"k={cross}: {mc:.3f}\nalready beats the baseline",
                    xy=(cross, mc), xytext=(cross * 1.6, mc - 0.085),
                    color=RED, fontsize=8.2, fontweight="bold",
                    arrowprops={"arrowstyle": "-", "color": RED, "lw": 1.0, "alpha": 0.6})

    ax.set_xscale("log", base=2)
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks], fontsize=7.6)
    ax.set_xlabel("normal images in the bank (k)", color=INK, fontsize=9)
    ax.set_ylabel("Image AUROC", color=INK, fontsize=9)
    ax.set_title("(a) Normal-image budget", color=INK, fontsize=9.8,
                 fontweight="bold", pad=8)

    # ---------------- (b, c) contamination ----------------------------------
    con = [r for r in res if r["part"] == "contam"]
    titles = {"synthetic": "(b) Contamination with synthetic anomalies",
              "real": "(c) Contamination with real anomalies"}
    for ax, source in zip(axes[1:], ("synthetic", "real"), strict=True):
        style(ax)
        rows = [r for r in con if r["source"] == source]
        if not rows:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, color=MUTED)
            ax.set_axis_off()
            continue
        rates = sorted({r["rate"] for r in rows})
        x = np.array(rates) * 100
        for clean, color, marker, mfc, label in (
            (False, RED, "o", RED, "raw bank"),
            (True, STEEL, "s", "white", "self-cleaning (drops 10 %)"),
        ):
            sel = [[r for r in rows if r["rate"] == q and r["clean"] == clean] for q in rates]
            if not all(sel):
                continue
            m = np.array([agg(s)[0] for s in sel])
            d = np.array([agg(s)[1] for s in sel])
            ax.fill_between(x, m - d, m + d, color=color, alpha=0.14, lw=0, zorder=2)
            ax.plot(x, m, color=color, lw=2.0, marker=marker, ms=5.5, mfc=mfc,
                    mec=color if clean else "white", mew=1.4, label=label, zorder=4)

        ax.axvline(BASE_RATE, color=MUTED, lw=1.0, ls=":", alpha=0.85, zorder=1)
        # y in axes fraction so the label survives the shared-ylim pass below;
        # anchored at the top, clear of the legend in the lower-left corner
        ax.text(BASE_RATE + 0.18, 0.97, f"MIIC's own rate ({BASE_RATE} %)",
                transform=ax.get_xaxis_transform(), color=MUTED, fontsize=7.4,
                va="top", rotation=90)
        ax.set_xlabel("unlabeled-pool contamination  ε (%)", color=INK, fontsize=9)
        ax.set_title(titles[source], color=INK, fontsize=9.8, fontweight="bold", pad=8)
        leg = ax.legend(frameon=False, fontsize=8.2, loc="lower left",
                        handlelength=1.8, borderaxespad=0.4)
        for t in leg.get_texts():
            t.set_color(INK)

    axes[1].set_ylabel("Image AUROC", color=INK, fontsize=9)
    # (b) and (c) answer the same question with different contaminants, so they
    # must share a y-range: different scales would exaggerate one of the drops.
    drawn = [ax for ax in axes[1:] if ax.axison]
    if len(drawn) == 2:
        lo = min(ax.get_ylim()[0] for ax in drawn)
        hi = max(ax.get_ylim()[1] for ax in drawn)
        for ax in drawn:
            ax.set_ylim(lo, hi)
    fig.tight_layout(w_pad=1.8)
    fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")  # vector, journal artwork
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
