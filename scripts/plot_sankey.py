#!/usr/bin/env python
"""Sankey diagram of the MIIC dataset composition (for the paper).

Flows: collection -> class (defect-free vs. defective) -> use (train / test).
The defective band is exaggerated for visibility (true share is 0.46%).

    uv run python scripts/plot_sankey.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path as MPath

OUT = Path("/autofs/space/crater_001/projects/micropath/semi/outputs/report/assets/sankey.png")

STEEL = "#5b7fa6"
RED = "#c0392b"
NODE = "#2b2f36"
INK = "#1a1c20"


def band(ax, x0, x1, y0, y1, color, alpha=0.5):
    """Filled flow from segment (x0, y0=(top,bot)) to (x1, y1=(top,bot))."""
    cx = (x0 + x1) / 2
    verts = [
        (x0, y0[0]), (cx, y0[0]), (cx, y1[0]), (x1, y1[0]),
        (x1, y1[1]), (cx, y1[1]), (cx, y0[1]), (x0, y0[1]),
        (x0, y0[0]),
    ]
    codes = [
        MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
        MPath.LINETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
        MPath.CLOSEPOLY,
    ]
    ax.add_patch(PathPatch(MPath(verts, codes), fc=color, ec="none", alpha=alpha))


def node(ax, x, y0, y1, w=12):
    ax.add_patch(Rectangle((x, y0), w, y1 - y0, fc=NODE, ec="none"))


def main() -> None:
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.set_xlim(0, 680)
    ax.set_ylim(315, 0)  # y increases downward
    ax.axis("off")

    xl, xm, xr, w = 96, 340, 560, 12

    # nodes (exaggerated defect band for visibility)
    node(ax, xl, 30, 289)               # MIIC
    node(ax, xm, 30, 280)               # sin defecto
    node(ax, xm, 295, 304)              # con defecto
    node(ax, xr, 30, 267)               # entrenamiento
    node(ax, xr, 279, 301)              # test

    # flows
    band(ax, xl + w, xm, (30, 280), (30, 280), STEEL)          # MIIC -> sin defecto
    band(ax, xl + w, xm, (280, 289), (295, 304), RED, 0.6)     # MIIC -> con defecto
    band(ax, xm + w, xr, (30, 267), (30, 267), STEEL)          # sin defecto -> train
    band(ax, xm + w, xr, (267, 280), (279, 292), STEEL)        # defect-free -> test
    band(ax, xm + w, xr, (295, 304), (292, 301), RED, 0.6)     # defective -> test

    def lbl(x, y, name, count, ha, color=INK, big=10.5):
        ax.text(x, y, name, ha=ha, va="center", fontsize=big, color=color, weight="bold")
        ax.text(x, y + 11, count, ha=ha, va="center", fontsize=9, color=color)

    lbl(xl + w / 2, 14, "MIIC", "25,276 images", "center")
    lbl(xm + w / 2, 14, "Defect-free", "25,160", "center")
    lbl(xm + w + 8, 299, "Defective", "116  (0.46 %)", "left", RED, 10)
    lbl(xr + w + 8, 143, "Training", "23,888 defect-free", "left")
    lbl(xr + w + 8, 284, "Real test", "1,272 defect-free + 116 defective", "left")

    fig.tight_layout()
    fig.savefig(OUT, dpi=140, bbox_inches="tight", facecolor="white")
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
