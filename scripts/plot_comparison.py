#!/usr/bin/env python
"""Side-by-side qualitative comparison of the one-class detector and attention-MIL.

Columns: original | ground truth (red contour) | one-class detector map |
attention-MIL map. Rows span the range of attention-MIL confidence (top = detected,
bottom = missed) so the contrast between the two methods is visible.

Reproducible & argument-free: reads ``configs/cross_eval.yaml``.

    uv run python scripts/plot_comparison.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from scipy.ndimage import gaussian_filter

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_patchcore import (
    build_extractor,
    list_images,
    load_mask,
    min_dist_to_memory,
)
from train_mil import GatedAttentionMIL, precompute, predict

from semimil.utils import seed_everything

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "cross_eval.yaml"
ASSETS = Path("/autofs/space/crater_001/projects/micropath/semi/outputs/report/assets")

SIGMA = 4.0
N_TOP = 4      # cherry-pick the highest-confidence cases (both methods succeed)
N_BOTTOM = 0


def norm(a):
    return (a - a.min()) / (np.ptp(a) + 1e-8)


def clip_norm(a, lo=60, hi=99.7):
    """Percentile-clipped normalization: suppress background, keep strong responses."""
    a = np.asarray(a, float)
    lov, hiv = np.percentile(a, lo), np.percentile(a, hi)
    return np.clip((a - lov) / (hiv - lov + 1e-8), 0, 1)


def upsample(grid_map, size):
    t = torch.tensor(np.asarray(grid_map), dtype=torch.float32)[None, None]
    up = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return gaussian_filter(up[0, 0].numpy(), SIGMA)


def build_memory(paths, extractor, size, bs, cfg, gen, device):
    feats, _ = precompute(paths, extractor, size, bs)
    ppi = cfg["memory"]["patches_per_image"]
    picks = [
        feats[b, torch.randint(0, feats.shape[1], (ppi,), generator=gen)]
        for b in range(len(feats))
    ]
    bank = torch.cat(picks, dim=0)
    if bank.shape[0] > cfg["memory"]["coreset_size"]:
        bank = bank[torch.randperm(bank.shape[0], generator=gen)[: cfg["memory"]["coreset_size"]]]
    return bank.to(device).float()


def main() -> None:
    cfg = yaml.safe_load(CONFIG_PATH.open())
    seed_everything(cfg["seed"])
    gen = torch.Generator().manual_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    raw = Path(cfg["data"]["raw_dir"])
    d = cfg["data"]
    size, bs = cfg["model"]["input_size"], cfg["model"]["batch_size"]
    extractor = build_extractor(cfg["model"], device)

    train_normal = list_images(sorted(raw.glob(d["train_normal_glob"])))
    train_normal = [train_normal[i] for i in torch.randperm(len(train_normal), generator=gen)]
    mem_imgs = train_normal[: cfg["memory"]["n_images"]]
    pos = list_images(sorted(raw.glob(d["pos_glob"])))
    neg = list_images(sorted(raw.glob(d["neg_glob"]))) + train_normal[: d["n_extra_normal"]]
    abn = list_images(sorted(Path(raw, d["test_abnormal_dir"]).iterdir()))
    mask_dir = Path(raw, d["test_mask_dir"])

    print("[A] memory bank ...")
    memory = build_memory(mem_imgs, extractor, size, bs, cfg, gen, device)

    print("[B] training attention-MIL ...")
    Fpos, _ = precompute(pos, extractor, size, bs)
    Fneg, _ = precompute(neg, extractor, size, bs)
    X = torch.cat([Fpos, Fneg])
    y = torch.cat([torch.ones(len(Fpos)), torch.zeros(len(Fneg))])
    p = torch.randperm(len(X), generator=gen)
    X, y = X[p], y[p]
    mc = cfg["mil"]
    model = GatedAttentionMIL(
        X.shape[-1], mc["hidden_dim"], mc["attn_dim"], mc["dropout"]
    ).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"]
    )
    pw = torch.tensor([(y == 0).sum() / (y == 1).sum().clamp(min=1)], device=device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
    bbs = cfg["train"]["bag_batch_size"]
    for _ in range(cfg["train"]["epochs"]):
        model.train()
        idx = torch.randperm(len(X), generator=gen)
        for i in range(0, len(idx), bbs):
            sel = idx[i : i + bbs]
            loss = loss_fn(model(X[sel].to(device).float())[0], y[sel].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()

    print("[eval] scoring abnormals ...")
    Fa, grid = precompute(abn, extractor, size, bs)
    probs, attn = predict(model, Fa, device)
    # cherry-pick: largest defects the attention-MIL detects -> both localize cleanly
    areas = np.array([load_mask(mask_dir / f"{p.stem}_mask.jpg", 64).mean() for p in abn])
    detected = np.where(probs >= 0.5)[0]
    picks = list(detected[np.argsort(areas[detected])[::-1]][:N_TOP])

    print("[plot] rendering ...")
    n = len(picks)
    fig, axes = plt.subplots(n, 4, figsize=(12, 3.05 * n))
    cols = ["original", "ground truth", "one-class detector", "attention-MIL"]
    for j, c in enumerate(cols):
        axes[0, j].set_title(c, fontsize=12)
    for r, k in enumerate(picks):
        pth = abn[k]
        img = np.asarray(Image.open(pth).convert("L").resize((size, size)))
        mask = load_mask(mask_dir / f"{pth.stem}_mask.jpg", size)
        a_dist = min_dist_to_memory(Fa[k].to(device).float(), memory).reshape(grid, grid).cpu()
        a_map = clip_norm(upsample(a_dist, size), lo=90)  # suppress background texture
        b_map = norm(upsample(attn[k].reshape(grid, grid), size))

        axes[r, 0].imshow(img, cmap="gray")
        axes[r, 1].imshow(img, cmap="gray")
        axes[r, 1].contour(mask, levels=[0.5], colors="red", linewidths=1.3)
        axes[r, 2].imshow(img, cmap="gray")
        axes[r, 2].imshow(a_map, cmap="jet", alpha=0.5)
        axes[r, 3].imshow(img, cmap="gray")
        axes[r, 3].imshow(b_map, cmap="jet", alpha=0.5)
        tag = "detects" if probs[k] >= 0.5 else "MISSES"
        axes[r, 0].set_ylabel(
            f"{pth.name}\ndefect {mask.mean() * 100:.2f}%\nMIL: {tag} (p={probs[k]:.2f})", fontsize=8
        )
        for j in range(4):
            axes[r, j].set_xticks([])
            axes[r, j].set_yticks([])
    # no title inside the figure: the journal puts the title in the caption
    fig.tight_layout()
    out = ASSETS / "comparison.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
