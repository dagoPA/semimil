#!/usr/bin/env python
"""Headline figure: sensitivity at specificity 0.99 on MIIC, all defects and small defects.

Two panels with one row per method (the proposed detector, the baseline detector and the
recent methods run under the same protocol): sensitivity at the 99th percentile of the
normal test scores over every defective image, and over the defects smaller than 0.1 % of
the image. Points are means over seeds; the 95 % CIs come from a stratified bootstrap of
the first seed (2,000 resamples), as in Table 7 and Appendix C.

Argument-free: reads configs/headline.yaml.

    uv run python scripts/plot_headline.py
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
CONFIG_PATH = REPO_ROOT / "configs" / "headline.yaml"

INK, MUTED, GRID = "#1d2330", "#5a6474", "#e5e8ee"
BLUE, GRAY, LIGHT = "#2b6cb0", "#6b7280", "#9aa3b1"
EXTS = {".jpg", ".jpeg", ".png"}


def images(d: Path):
    return sorted(p for p in d.iterdir() if p.suffix.lower() in EXTS)


def sens_at_spec(y, s, small, spec):
    """Sensitivity over all defective images and over the small ones, threshold on normals."""
    thr = np.percentile(s[y == 0], 100 * spec)
    pos = s > thr
    return pos[y == 1].mean(), pos[(y == 1) & small].mean()


def runs(m, cfg, names):
    out = Path(cfg["outputs"])
    if m["kind"] == "proposed":
        zs = []
        for run in m["runs"]:
            z = np.load(out / run / "raw" / "n23688.npz")
            t, c = z["test_map"].astype(np.float32), z["calib_map"].astype(np.float32)
            ti, ci = t.reshape(len(t), -1).max(1), c.reshape(len(c), -1).max(1)
            zs.append((ti - ci.mean()) / ci.std())
        return [np.mean(zs, 0)]
    if m["kind"] == "baseline":
        return [np.load(out / f)["test_score"].astype(np.float64) for f in m["files"]]
    scores = []
    for d in sorted((out / "sota" / m["path"]).glob("seed*")):
        sc = {r["name"]: float(r["score"]) for r in csv.DictReader(open(d / "image_scores.csv"))}
        scores.append(np.array([sc[n] for n in names]))
    return scores


def main() -> None:
    cfg = yaml.safe_load(CONFIG_PATH.open())
    raw = Path(cfg["data"]["raw_dir"])
    d = cfg["data"]
    norm, abn = images(raw / d["test_normal_dir"]), images(raw / d["test_abnormal_dir"])
    names = [p.name for p in norm] + [p.name for p in abn]
    y = np.r_[np.zeros(len(norm), int), np.ones(len(abn), int)]
    area = np.r_[np.zeros(len(norm)), [
        (np.asarray(Image.open(raw / d["test_mask_dir"] / f"{p.stem}_mask.jpg").convert("L")) > 127).mean() * 100
        for p in abn]]
    small = (y == 1) & (area < cfg["small_area_pct"])
    spec, B = cfg["spec"], cfg["bootstrap"]

    rows = []
    for m in cfg["methods"]:
        rs = runs(m, cfg, names)
        point = np.mean([sens_at_spec(y, s, small, spec) for s in rs], 0)
        rng = np.random.RandomState(cfg["seed"])
        i0, i1 = np.where(y == 0)[0], np.where(y == 1)[0]
        boot = []
        for _ in range(B):
            idx = np.r_[rng.choice(i0, len(i0)), rng.choice(i1, len(i1))]
            boot.append(sens_at_spec(y[idx], rs[0][idx], small[idx], spec))
        lo, hi = np.percentile(boot, 2.5, 0), np.percentile(boot, 97.5, 0)
        rows.append((m["label"], m["kind"], point, lo, hi, len(rs)))
        print(f"{m['label']:18s} seeds={len(rs)} all={point[0]:.3f} [{lo[0]:.3f}, {hi[0]:.3f}] "
              f"small={point[1]:.3f} [{lo[1]:.3f}, {hi[1]:.3f}]")
    rows.sort(key=lambda r: r[2][0])  # bottom to top: best on all defects at the top

    plt.rcParams.update({"font.family": "Nimbus Sans", "font.size": 7.5})
    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.45), sharey=True)  # text width of the preprint
    titles = (f"All defective images ({int(y.sum())})",
              f"Defects below {cfg['small_area_pct']:g} % of the image ({int(small.sum())})")
    for j, ax in enumerate(axes):
        for k, (label, kind, point, lo, hi, _) in enumerate(rows):
            col = BLUE if kind == "proposed" else (LIGHT if kind == "baseline" else GRAY)
            lw, ms = (2.2, 7) if kind == "proposed" else (1.5, 5.5)
            ax.plot([lo[j], hi[j]], [k, k], color=col, lw=lw, solid_capstyle="round", zorder=2)
            ax.plot(point[j], k, "o", color=col, ms=ms, mec="white", mew=1.0, zorder=3)
            ax.text(hi[j] + 0.02, k, f"{point[j]:.2f}", va="center", ha="left", fontsize=7,
                    color=BLUE if kind == "proposed" else INK,
                    fontweight="bold" if kind == "proposed" else "normal")
        ax.set_title(titles[j], fontsize=7.8, color=INK, loc="left", pad=5)
        ax.set_xlim(0.0 if j else 0.5, 1.1)
        ax.set_xticks(np.arange(0.0, 1.01, 0.2) if j else np.arange(0.5, 1.01, 0.1))
        ax.set_xlabel("Sensitivity at specificity 0.99", color=INK, fontsize=7.5)
        ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
        ax.tick_params(colors=MUTED, length=0, labelsize=7)
        for s_ in ("top", "right", "left"):
            ax.spines[s_].set_visible(False)
        ax.spines["bottom"].set_color(MUTED), ax.spines["bottom"].set_linewidth(0.6)
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels([r[0] for r in rows])
    for t, r in zip(axes[0].get_yticklabels(), rows):
        if r[1] == "proposed":
            t.set_color(BLUE), t.set_fontweight("bold")
        else:
            t.set_color(INK)
    fig.tight_layout(w_pad=1.5)
    out = Path(cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(Path(cfg["outputs"]) / "headline_preview.png", dpi=250)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
