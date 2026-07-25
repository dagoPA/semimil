#!/usr/bin/env python
"""PaDiM baseline (Defard et al., ICPR-W 2021) on MIIC.

Models each feature-map position with a multivariate Gaussian over pretrained
CNN features and scores by Mahalanobis distance. Unlike the memory-bank detector
(Track A), PaDiM assumes rough spatial alignment across images -- a poor fit for
IC microscopy, which this baseline quantifies. Multi-seed (the seed drives the
random channel subset and the training sample).

Reproducible & argument-free: reads ``configs/padim.yaml``.
    CUDA_VISIBLE_DEVICES=1 uv run python scripts/run_padim.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_patchcore import batched, build_extractor, list_images, load_image, load_mask

from semimil.utils import seed_everything

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "padim.yaml"


@torch.no_grad()
def extract(paths, extractor, size, bs, dim_idx):
    """-> ([n, P, d] features, grid) where d = len(dim_idx)."""
    out, grid = [], None
    for batch in batched(paths, bs):
        x = torch.stack([load_image(p, size) for p in batch])
        f = extractor(x)  # [B, Hf, Wf, D]
        grid = f.shape[1]
        f = f.reshape(f.shape[0], -1, f.shape[-1])[:, :, dim_idx]
        out.append(f.cpu())
    return torch.cat(out), grid


def run_seed(cfg, extractor, seed, train_normal, test_norm, test_abn, y, device):
    m = cfg["model"]
    gen = torch.Generator().manual_seed(seed)
    seed_everything(seed)

    # probe feature dim, then pick a random channel subset
    probe = extractor(torch.stack([load_image(train_normal[0], m["input_size"])]))
    d_full = probe.shape[-1]
    dim_idx = torch.randperm(d_full, generator=gen)[: m["n_dims"]]

    # fit per-position Gaussians on normal images
    idx = torch.randperm(len(train_normal), generator=gen)[: m["n_train"]]
    tr_paths = [train_normal[i] for i in idx]
    Ftr, grid = extract(tr_paths, extractor, m["input_size"], m["batch_size"], dim_idx)
    Ftr = Ftr.to(device).float()  # [N, P, d]
    n, _, d = Ftr.shape
    mu = Ftr.mean(0)  # [P, d]
    cen = Ftr - mu
    cov = torch.einsum("npd,npe->pde", cen, cen) / (n - 1)
    cov += m["eps"] * torch.eye(d, device=device)
    cov_inv = torch.linalg.inv(cov)  # [P, d, d]
    del Ftr, cen, cov

    def score_maps(paths):
        maps = []
        Ft, _ = extract(paths, extractor, m["input_size"], m["batch_size"], dim_idx)
        for i in range(Ft.shape[0]):
            diff = Ft[i].to(device).float() - mu  # [P, d]
            d2 = torch.einsum("pd,pde,pe->p", diff, cov_inv, diff)
            maps.append(d2.clamp(min=0).sqrt().reshape(grid, grid).cpu())
        return maps

    nmaps = score_maps(test_norm)
    amaps = score_maps(test_abn)
    img_scores = np.array([float(mm.max()) for mm in nmaps + amaps])

    ms = cfg["eval"]["map_size"]

    def up(mm):
        t = mm[None, None]
        return F.interpolate(t, size=(ms, ms), mode="bilinear", align_corners=False)[0, 0]

    mask_dir = Path(cfg["data"]["raw_dir"], cfg["data"]["test_mask_dir"])
    px_pred, px_true = [], []
    for mm in nmaps:
        px_pred.append(up(mm).numpy().reshape(-1))
        px_true.append(np.zeros(ms * ms, np.uint8))
    for mm, pth in zip(amaps, test_abn, strict=True):
        px_pred.append(up(mm).numpy().reshape(-1))
        px_true.append(load_mask(mask_dir / f"{pth.stem}_mask.jpg", ms).reshape(-1))
    px_pred, px_true = np.concatenate(px_pred), np.concatenate(px_true)

    rec = {
        "seed": seed,
        "image_auroc": float(roc_auc_score(y, img_scores)),
        "image_auprc": float(average_precision_score(y, img_scores)),
        "pixel_auroc": float(roc_auc_score(px_true, px_pred)),
        "pixel_auprc": float(average_precision_score(px_true, px_pred)),
    }
    return rec, img_scores


def main() -> None:
    cfg = yaml.safe_load(CONFIG_PATH.open())
    dev = cfg["device"]
    device = ("cuda" if torch.cuda.is_available() else "cpu") if dev == "auto" else dev
    raw = Path(cfg["data"]["raw_dir"])
    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    train_normal = list_images(sorted(raw.glob(cfg["data"]["train_normal_glob"])))
    test_norm = list_images(sorted(Path(raw, cfg["data"]["test_normal_dir"]).iterdir()))
    test_abn = list_images(sorted(Path(raw, cfg["data"]["test_abnormal_dir"]).iterdir()))
    y = np.array([0] * len(test_norm) + [1] * len(test_abn))
    print(f"device={device}  test={len(test_norm)}N/{len(test_abn)}A", flush=True)

    extractor = build_extractor(cfg["model"], device)
    names = [p.name for p in test_norm + test_abn]
    results = []
    for i, seed in enumerate(cfg["seeds"]):
        r, scores = run_seed(cfg, extractor, seed, train_normal, test_norm, test_abn, y, device)
        results.append(r)
        print(
            f"seed {seed}: image AUROC {r['image_auroc']:.4f} AUPRC {r['image_auprc']:.4f} | "
            f"pixel AUROC {r['pixel_auroc']:.4f} AUPRC {r['pixel_auprc']:.4f}",
            flush=True,
        )
        (out_dir / "results.json").write_text(json.dumps(results, indent=2))
        if i == 0:  # save per-image scores (seed 0) for the DeLong test
            import csv

            with (out_dir / "image_scores.csv").open("w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["name", "label", "score"])
                for nm, yl, sc in zip(names, y, scores, strict=True):
                    w.writerow([nm, int(yl), float(sc)])

    a = np.array([r["image_auroc"] for r in results])
    pr = np.array([r["image_auprc"] for r in results])
    print(f"\nPaDiM AUROC {a.mean():.3f}+-{a.std():.3f} AUPRC {pr.mean():.3f}+-{pr.std():.3f}")
    print(f"saved -> {out_dir}")


if __name__ == "__main__":
    main()
