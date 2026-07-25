#!/usr/bin/env python
"""Forest plot of the method ranking with 95% bootstrap CIs (image AUROC).

Reads the saved bootstrap + DeLong results and renders a ranked point-and-error-
bar figure. Identity is encoded by BOTH color and marker shape (filled circle =
our methods, open square = baselines), so it never relies on color alone; a
reference band marks A-DINOv2's CI to show which methods overlap it.

    uv run python scripts/plot_ranking.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

STATS = Path("/autofs/space/crater_001/projects/micropath/semi/outputs/stats")
OUT = Path("/autofs/space/crater_001/projects/micropath/semi/outputs/report/assets/ranking.png")

RED = "#cf4128"      # our methods
STEEL = "#2f6a9c"    # baselines / literature
INK = "#1a1c20"
MUTED = "#566070"
GRID = "#e4e7ec"
REF = "A-DINOv2 (one-class)"

OURS = {"A-DINOv2 (one-class)", "B-attn-MIL", "A-WRN50 (one-class)", "A-DINOv2-dominio"}
LABEL = {
    "A-DINOv2 (one-class)": "DINOv2 · one-class  (nuestro)",
    "B-attn-MIL": "Attention-MIL  (nuestro)",
    "A-WRN50 (one-class)": "WRN50 · one-class  (nuestro)",
    "A-DINOv2-dominio": "DINOv2-dominio  (nuestro)",
    "FastFlow": "FastFlow ‡",
    "PaDiM": "PaDiM",
    "STFPM": "STFPM ‡",
    "CFA": "CFA ‡",
    "DRAEM": "DRAEM",
}


def main() -> None:
    boot = {r["method"]: r for r in json.loads((STATS / "bootstrap.json").read_text())["results"]}
    delong = json.loads((STATS / "delong.json").read_text())
    pmap = {c["method"]: c["p_delong"] for c in delong["comparisons"]}

    rows = sorted(boot.values(), key=lambda r: r["auroc"], reverse=True)
    ref = boot[REF]

    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    ax.axvspan(ref["auroc_ci"][0], ref["auroc_ci"][1], color=RED, alpha=0.06, zorder=0)
    ax.axvline(ref["auroc"], color=RED, lw=1, ls="--", alpha=0.5, zorder=1)

    n = len(rows)
    for i, r in enumerate(rows):
        y = n - 1 - i
        m = r["method"]
        is_ours = m in OURS
        color = RED if is_ours else STEEL
        lo, hi = r["auroc_ci"]
        ax.plot([lo, hi], [y, y], color=color, lw=2, zorder=2, solid_capstyle="round")
        if is_ours:
            ax.plot(r["auroc"], y, "o", color=color, ms=10 if m == REF else 8,
                    mec="white", mew=1.2, zorder=3)
        else:
            ax.plot(r["auroc"], y, "s", color="white", ms=8, mec=color, mew=2, zorder=3)
        ax.text(hi + 0.006, y, f"{r['auroc']:.3f}", va="center", ha="left",
                fontsize=8.5, color=INK, fontweight="bold" if m == REF else "normal")
        # significance vs reference
        p = pmap.get(m)
        tag = "ref" if m == REF else ("n.s." if (p is not None and p >= 0.05) else "")
        if tag:
            ax.text(lo - 0.006, y, tag, va="center", ha="right", fontsize=7.5,
                    color=MUTED, style="italic")

    ax.set_yticks(range(n))
    ax.set_yticklabels([LABEL[r["method"]] for r in rows][::-1], fontsize=9)
    ax.set_xlabel("AUROC de imagen  (IC 95% por bootstrap, 2000 remuestreos)", fontsize=9.5)
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
        Line2D([0], [0], marker="o", color=RED, lw=0, mfc=RED, mec="white", ms=9,
               label="Nuestros métodos"),
        Line2D([0], [0], marker="s", color=STEEL, lw=0, mfc="white", mec=STEEL, mew=2, ms=8,
               label="Baselines / literatura  (‡ reproducido)"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=8.5)
    ax.set_title("Ranking de métodos en el test real (n=1388, 116 con defecto)",
                 fontsize=11, color=INK, pad=10)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
