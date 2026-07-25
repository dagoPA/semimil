#!/usr/bin/env python
"""Plot paired examples: original image | anomaly heatmap | overlay | GT mask.

Selects the highest- and lowest-scoring abnormal test images (clear detections
vs hard cases) from a previous ``run_patchcore.py`` run, recomputes their
anomaly maps, and saves a 4-column figure.

Reproducible & argument-free: reads ``configs/patchcore.yaml``.

    uv run python scripts/plot_heatmaps.py
"""

from __future__ import annotations

import csv
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

# Reuse the pipeline machinery without duplicating it.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_patchcore import (
    batched,
    build_extractor,
    list_images,
    load_image,
    load_mask,
    min_dist_to_memory,
)

from semimil.utils import seed_everything

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "patchcore.yaml"


def build_memory(extractor, raw, cfg, size, bs, rng):
    train_normals = list_images(sorted(raw.glob(cfg["data"]["train_normal_glob"])))
    rng.shuffle(train_normals)
    mem_imgs = train_normals[: cfg["memory"]["n_images"]]
    ppi = cfg["memory"]["patches_per_image"]
    bank = []
    for batch in batched(mem_imgs, bs):
        x = torch.stack([load_image(p, size) for p in batch])
        feats = extractor(x)
        feats = feats.reshape(feats.shape[0], -1, feats.shape[-1])
        for b in range(feats.shape[0]):
            idx = torch.randint(0, feats.shape[1], (ppi,))
            bank.append(feats[b, idx].cpu())
    bank_t = torch.cat(bank, dim=0)
    if bank_t.shape[0] > cfg["memory"]["coreset_size"]:
        sel = torch.randperm(bank_t.shape[0])[: cfg["memory"]["coreset_size"]]
        bank_t = bank_t[sel]
    return bank_t


def anomaly_map(extractor, path, memory, size, sigma):
    x = load_image(path, size)[None]
    feats = extractor(x)[0]  # [Hf,Wf,D]
    hf, wf, d = feats.shape
    dist = min_dist_to_memory(feats.reshape(-1, d), memory).reshape(hf, wf)
    up = F.interpolate(dist[None, None], size=(size, size), mode="bilinear", align_corners=False)
    return gaussian_filter(up[0, 0].cpu().numpy(), sigma)


def select_examples(csv_path, n_top, n_bottom):
    rows = [r for r in csv.DictReader(csv_path.open()) if r["label"] == "1"]
    rows.sort(key=lambda r: float(r["score"]))
    top = rows[-n_top:][::-1]
    bottom = rows[:n_bottom]
    return [("top", r) for r in top] + [("bottom", r) for r in bottom]


def main() -> None:
    cfg = yaml.safe_load(CONFIG_PATH.open())
    seed_everything(cfg["seed"])
    import random

    rng = random.Random(cfg["seed"])
    device = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if cfg["device"] == "auto"
        else cfg["device"]
    )
    raw = Path(cfg["data"]["raw_dir"])
    size = cfg["model"]["input_size"]
    bs = cfg["model"]["batch_size"]
    sigma = cfg["score"]["gaussian_sigma"]
    out_dir = Path(cfg["output"]["dir"]) / cfg["run_name"]
    abn_dir = Path(raw, cfg["data"]["test_abnormal_dir"])
    mask_dir = Path(raw, cfg["data"]["test_mask_dir"])

    extractor = build_extractor(cfg["model"], device)
    print("[memory] rebuilding bank ...")
    memory = build_memory(extractor, raw, cfg, size, bs, rng).to(device).float()

    picks = select_examples(
        out_dir / "image_scores.csv", cfg["plot"]["n_top"], cfg["plot"]["n_bottom"]
    )
    n = len(picks)
    alpha = cfg["plot"]["overlay_alpha"]
    fig, axes = plt.subplots(n, 4, figsize=(12, 3 * n))
    cols = ["original", "heatmap", "overlay", "GT mask"]
    for j, c in enumerate(cols):
        axes[0, j].set_title(c, fontsize=12)

    for i, (kind, r) in enumerate(picks):
        name = r["name"]
        path = abn_dir / name
        img = np.asarray(Image.open(path).convert("L").resize((size, size)))
        heat = anomaly_map(extractor, path, memory, size, sigma)
        heat_n = (heat - heat.min()) / (np.ptp(heat) + 1e-8)
        mask = load_mask(mask_dir / f"{path.stem}_mask.jpg", size) * 255

        axes[i, 0].imshow(img, cmap="gray")
        axes[i, 1].imshow(heat_n, cmap="jet")
        axes[i, 2].imshow(img, cmap="gray")
        axes[i, 2].imshow(heat_n, cmap="jet", alpha=alpha)
        axes[i, 3].imshow(mask, cmap="gray")
        axes[i, 0].set_ylabel(f"{kind}\n{name}\nscore={float(r['score']):.1f}", fontsize=9)
        for j in range(4):
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])

    fig.suptitle(
        f"PatchCore [{cfg['run_name']}] — top (clear) vs bottom (hard) abnormals", y=1.0
    )
    fig.tight_layout()
    out = out_dir / "heatmaps_examples.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
