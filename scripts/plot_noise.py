#!/usr/bin/env python
"""Robustness to image noise on MIIC: numbers and figure.

Reads the runs of scripts/run_noise.py (Gaussian noise of sigma gray levels on every
test and calibration image, with the clean memory bank or with a memory bank rebuilt from
images at the same sigma) and, for sigma = 0, the clean runs of the main text. For each
setting it computes image AUROC and the sensitivity at specificity 0.99 over all defective
images and over the defects below 0.1 % of the image: the proposed detector is the fused
z-score of its two representations, standardized on the (equally noisy) calibration
images; the baseline detector is the mean over its three memory-bank seeds. 95 % CIs come
from a stratified bootstrap (2,000 resamples) of the proposed detector and of the first
seed of the baseline detector.

The figure shows (a) one small defect at every sigma and (b, c) the sensitivity at
specificity 0.99 against sigma, with the clean memory bank (solid) and the rebuilt memory
bank (dashed). Writes outputs/noise/summary.json and the figure.

Argument-free: reads configs/noise_plot.yaml.

    uv run python scripts/plot_noise.py
"""

from __future__ import annotations

import json
import zlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image
from scipy import ndimage as ndi
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "noise_plot.yaml"

INK, MUTED, GRID = "#1d2330", "#5a6474", "#e5e8ee"
BLUE, GRAY = "#2b6cb0", "#6b7280"
EXTS = {".jpg", ".jpeg", ".png"}


def images(d: Path):
    return sorted(p for p in d.iterdir() if p.suffix.lower() in EXTS)


def fused(files):
    """Fused z-score of the proposed detector from its two representations."""
    zs = []
    for f in files:
        z = np.load(f)
        t, c = z["test_map"].astype(np.float32), z["calib_map"].astype(np.float32)
        ti, ci = t.reshape(len(t), -1).max(1), c.reshape(len(c), -1).max(1)
        zs.append((ti - ci.mean()) / ci.std())
    return np.mean(zs, 0).astype(np.float64)


def metrics(y, s, small, spec):
    thr = np.percentile(s[y == 0], 100 * spec)
    pos = s >= thr
    return np.array([roc_auc_score(y, s), pos[y == 1].mean(), pos[small].mean()])


def boot(y, s, small, spec, B, seed):
    rng = np.random.RandomState(seed)
    i0, i1 = np.where(y == 0)[0], np.where(y == 1)[0]
    v = [metrics(y[idx], s[idx], small[idx], spec)
         for idx in (np.r_[rng.choice(i0, len(i0)), rng.choice(i1, len(i1))] for _ in range(B))]
    return np.percentile(v, 2.5, 0), np.percentile(v, 97.5, 0)


def noisy(path: Path, sigma: float) -> np.ndarray:
    """The image as scripts/run_noise.py sees it before resizing."""
    a = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    if sigma > 0:
        rng = np.random.default_rng([int(round(sigma * 100)), zlib.crc32(path.name.encode())])
        a = np.clip(np.rint(a + rng.normal(0.0, sigma, a.shape)), 0, 255)
    return a.astype(np.uint8)


def main() -> None:
    cfg = yaml.safe_load(CONFIG_PATH.open())
    d, raw = cfg["data"], Path(cfg["data"]["raw_dir"])
    norm, abn = images(raw / d["test_normal_dir"]), images(raw / d["test_abnormal_dir"])
    y = np.r_[np.zeros(len(norm), int), np.ones(len(abn), int)]
    masks = [np.asarray(Image.open(raw / d["test_mask_dir"] / f"{p.stem}_mask.jpg").convert("L")) > 127
             for p in abn]
    area = np.r_[np.zeros(len(norm)), [m.mean() * 100 for m in masks]]
    small = (y == 1) & (area < cfg["small_area_pct"])
    spec, B, seed = cfg["spec"], cfg["bootstrap"], cfg["seed"]
    noise, clean = Path(cfg["noise_dir"]) / "raw", Path(cfg["outputs"])

    res = {"sigmas": [0] + cfg["sigmas"], "settings": {}}
    for setting in ("clean_bank", "matched_bank"):
        rows = {}
        for sigma in [0] + cfg["sigmas"]:
            b = 0 if setting == "clean_bank" else sigma
            if sigma == 0:
                prop = fused([clean / r / "raw" / "n23688.npz" for r in cfg["proposed_clean"]])
                base = [np.load(clean / f.format(k))["test_score"].astype(np.float64)
                        for k, f in enumerate([cfg["baseline_clean"]] * 3)]
            else:
                pf = [noise / f"{r}_bank{b:g}_s{sigma:g}.npz" for r in ("L8", "L10_pool3")]
                bf = [noise / f"baseline_bank{b:g}_s{sigma:g}_seed{k}.npz" for k in range(3)]
                if not all(f.exists() for f in pf + bf):
                    print(f"{setting} sigma {sigma}: missing runs, skipped")
                    continue
                prop = fused(pf)
                base = [np.load(f)["test_score"].astype(np.float64) for f in bf]
            mp = metrics(y, prop, small, spec)
            mb = np.mean([metrics(y, s, small, spec) for s in base], 0)
            lp, hp = boot(y, prop, small, spec, B, seed)
            lb, hb = boot(y, base[0], small, spec, B, seed)
            rows[sigma] = {"proposed": mp.tolist(), "proposed_ci": [lp.tolist(), hp.tolist()],
                           "baseline": mb.tolist(), "baseline_ci_seed0": [lb.tolist(), hb.tolist()]}
            print(f"{setting:12s} sigma {sigma:>2}: proposed AUROC {mp[0]:.3f} sens {mp[1]:.3f} "
                  f"[{lp[1]:.3f}, {hp[1]:.3f}] small {mp[2]:.3f} [{lp[2]:.3f}, {hp[2]:.3f}] | "
                  f"baseline AUROC {mb[0]:.3f} sens {mb[1]:.3f} small {mb[2]:.3f}")
        res["settings"][setting] = rows
    res["columns"] = ["auroc", "sens", "sens_small"]
    res["n_defective"], res["n_small"] = int(y.sum()), int(small.sum())
    out_json = Path(cfg["noise_dir"]) / "summary.json"
    out_json.write_text(json.dumps(res, indent=2))

    # example: of the small defects whose crop fits inside the image, the one of median area
    h = cfg["crop"] // 2
    centers = {i: tuple(int(round(v)) for v in ndi.center_of_mass(masks[i])) for i in range(len(abn))
               if area[len(norm) + i] < cfg["small_area_pct"]}
    inside = [i for i, (cy, cx) in centers.items()
              if h <= cy <= masks[i].shape[0] - h and h <= cx <= masks[i].shape[1] - h]
    pick = sorted(inside, key=lambda i: area[len(norm) + i])[len(inside) // 2]
    cy, cx = centers[pick]
    y0, x0 = cy - h, cx - h

    plt.rcParams.update({"font.family": "Nimbus Sans", "font.size": 7.5})
    fig = plt.figure(figsize=(6.4, 2.6))
    gi = fig.add_gridspec(2, 2, wspace=0.06, hspace=0.22, left=0.01, right=0.205, top=0.86, bottom=0.3)
    gp = fig.add_gridspec(1, 2, wspace=0.22, left=0.3, right=0.99, top=0.86, bottom=0.3)
    sig = [0] + cfg["sigmas"]
    for k, s in enumerate(sig):
        ax = fig.add_subplot(gi[k // 2, k % 2])
        ax.imshow(noisy(abn[pick], s)[y0 : y0 + 2 * h, x0 : x0 + 2 * h], cmap="gray", vmin=0, vmax=255)
        ax.contour(masks[pick][y0 : y0 + 2 * h, x0 : x0 + 2 * h], levels=[0.5], colors="#2f9e44",
                   linewidths=0.6)
        ax.set_title(f"$\\sigma$ = {s}", fontsize=7, color=INK, pad=2)
        ax.set_xticks([]), ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.text(0.01, 0.95, "(a)", fontsize=8, color=INK, fontweight="bold")
    titles = (f"(b) All defective images ({int(y.sum())})",
              f"(c) Defects below {cfg['small_area_pct']:g} % of the image ({int(small.sum())})")
    for j, col in enumerate((1, 2)):
        ax = fig.add_subplot(gp[0, j])
        for setting, ls in (("clean_bank", "-"), ("matched_bank", "--")):
            rows = res["settings"][setting]
            xs = [s for s in sig if s in rows]
            for key, colr, lab in (("proposed", BLUE, "Proposed"), ("baseline", GRAY, "Baseline")):
                ys = [rows[s][key][col] for s in xs]
                ax.plot(xs, ys, ls, color=colr, lw=1.6 if key == "proposed" else 1.2,
                        marker="o", ms=3.5, mec="white", mew=0.6,
                        label=f"{lab}, {'clean' if setting == 'clean_bank' else 'noisy'} memory bank")
        ax.set_title(titles[j], fontsize=7.6, color=INK, loc="left", pad=4)
        ax.set_xticks(sig)
        ax.set_xlabel("Noise $\\sigma$ (gray levels)", color=INK)
        ax.set_ylim(0.0 if j else 0.5, 1.0)
        if j == 0:
            ax.set_ylabel("Sensitivity at specificity 0.99", color=INK)
        ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
        ax.tick_params(colors=MUTED, length=0, labelsize=7)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        for s_ in ("left", "bottom"):
            ax.spines[s_].set_color(MUTED), ax.spines[s_].set_linewidth(0.6)
    handles, labels = fig.axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=6.8,
               bbox_to_anchor=(0.645, 0.0), handlelength=2.6, columnspacing=2.0)
    out = Path(cfg["output"])
    fig.savefig(out)
    fig.savefig(Path(cfg["noise_dir"]) / "noise_preview.png", dpi=250)
    print(f"wrote {out_json} and {out} (example {abn[pick].name}, area {area[len(norm) + pick]:.3f} %)")


if __name__ == "__main__":
    main()
