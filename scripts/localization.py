#!/usr/bin/env python
"""Pixel-level localization metrics (pixel AUROC + PRO) for the one-class methods.

Reproduces each Track A run from the config saved in its metrics.json, rebuilds
the memory bank, and scores the real test set, upsampling anomaly maps to a common
resolution so pixel AUROC and PRO are computed uniformly. PRO (per-region overlap,
integrated up to 30% FPR) weights every ground-truth region equally regardless of
size -- the standard localization metric that plain pixel AUROC hides.

    uv run python scripts/localization.py
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import label
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_patchcore import (  # noqa: E402
    batched,
    build_extractor,
    list_images,
    load_image,
    load_mask,
    min_dist_to_memory,
)

from semimil.utils import seed_everything  # noqa: E402

ROOT = Path("/autofs/space/crater_001/projects/micropath/semi")
OUT = ROOT / "outputs/localization"
MAP = 256          # common map resolution for pixel AUROC + PRO
RUNS = ["dinov2", "wrn50", "dinov2_domain"]
LABEL = {"dinov2": "A-DINOv2 (one-class)", "wrn50": "A-WRN50 (one-class)",
         "dinov2_domain": "A-DINOv2-dominio"}


def pro_auc(maps: np.ndarray, masks: np.ndarray, max_fpr: float = 0.3,
            n_th: int = 256) -> float:
    """Per-region-overlap (PRO) AUC integrated over FPR in [0, max_fpr], in [0,1].

    maps  [N,H,W] float anomaly scores; masks [N,H,W] binary ground truth. PRO
    weights every ground-truth region equally regardless of size.
    """
    gt = masks.astype(bool)
    if not gt.any():
        return float("nan")
    neg = np.sort(maps[~gt].ravel())          # non-defect pixel scores, ascending
    n_neg = neg.size
    regions = []                               # sorted scores inside each GT region
    for i in range(maps.shape[0]):
        if not gt[i].any():
            continue
        lab, k = label(gt[i])
        for r in range(1, k + 1):
            regions.append(np.sort(maps[i][lab == r]))

    # thresholds chosen so FPR is densely and evenly sampled across [0, max_fpr]:
    # the threshold achieving a target FPR f is the (1-f) quantile of neg scores.
    tf = np.linspace(0.0, max_fpr, n_th)
    th = np.quantile(neg, np.clip(1.0 - tf, 0.0, 1.0))
    fprs = np.array([1.0 - np.searchsorted(neg, t, "left") / n_neg for t in th])
    pros = np.array([np.mean([1.0 - np.searchsorted(rs, t, "left") / rs.size
                              for rs in regions]) for t in th])

    order = np.argsort(fprs)
    fprs, pros = fprs[order], pros[order]
    # anchor at the origin (infinite threshold -> no pixels flagged)
    if fprs[0] > 0:
        fprs = np.concatenate([[0.0], fprs])
        pros = np.concatenate([[0.0], pros])
    # cap exactly at max_fpr by interpolation so the domain is [0, max_fpr]
    if fprs[-1] > max_fpr:
        p_cap = float(np.interp(max_fpr, fprs, pros))
        keep = fprs < max_fpr
        fprs = np.concatenate([fprs[keep], [max_fpr]])
        pros = np.concatenate([pros[keep], [p_cap]])
    trapz = getattr(np, "trapezoid", None) or np.trapz  # np.trapz gone in numpy 2
    return float(trapz(pros, fprs) / max_fpr)


def maps_for_run(run: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(normal_maps, abn_maps, abn_masks) at MAP resolution + image labels/scores."""
    cfg = json.loads((ROOT / f"outputs/patchcore/{run}/metrics.json").read_text())["config"]
    seed_everything(cfg["seed"])
    rng = random.Random(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    raw = Path(cfg["data"]["raw_dir"])
    size = cfg["model"]["input_size"]
    bs = cfg["model"]["batch_size"]
    extractor = build_extractor(cfg["model"], device)

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
    bank_t = torch.cat(bank, 0)
    if bank_t.shape[0] > cfg["memory"]["coreset_size"]:
        sel = torch.randperm(bank_t.shape[0])[: cfg["memory"]["coreset_size"]]
        bank_t = bank_t[sel]
    memory = bank_t.to(device).float()

    test_normals = list_images(sorted(Path(raw, cfg["data"]["test_normal_dir"]).iterdir()))
    test_abn = list_images(sorted(Path(raw, cfg["data"]["test_abnormal_dir"]).iterdir()))
    mask_dir = Path(raw, cfg["data"]["test_mask_dir"])
    top_k = cfg["score"]["top_k"]

    def score(paths, abn):
        maps, masks, iscore = [], [], []
        for batch in batched(paths, bs):
            x = torch.stack([load_image(p, size) for p in batch])
            feats = extractor(x)
            hf, wf, d = feats.shape[1:]
            for b, p in enumerate(batch):
                dist = min_dist_to_memory(feats[b].reshape(-1, d), memory)
                amap = dist.reshape(hf, wf)
                iscore.append(torch.topk(dist, top_k).values.mean().item())
                up = F.interpolate(amap[None, None], size=(MAP, MAP), mode="bilinear",
                                   align_corners=False)[0, 0].cpu().numpy()
                maps.append(up)
                masks.append(load_mask(mask_dir / f"{p.stem}_mask.jpg", MAP) if abn
                             else np.zeros((MAP, MAP), np.uint8))
        return maps, masks, iscore

    nmaps, nmasks, ns = score(test_normals, False)
    amaps, amasks, as_ = score(test_abn, True)
    all_maps = np.stack(nmaps + amaps)
    all_masks = np.stack(nmasks + amasks)
    y_img = np.array([0] * len(ns) + [1] * len(as_))
    s_img = np.array(ns + as_)
    return all_maps, all_masks, y_img, s_img


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for run in RUNS:
        print(f"\n=== {run} ===", flush=True)
        t0 = time.time()
        maps, masks, y_img, s_img = maps_for_run(run)
        px_auroc = float(roc_auc_score(masks.ravel(), maps.ravel()))
        px_auprc = float(average_precision_score(masks.ravel(), maps.ravel()))
        pro = pro_auc(maps, masks)
        img_auroc = float(roc_auc_score(y_img, s_img))
        r = {"method": LABEL[run], "run": run, "image_auroc": img_auroc,
             "pixel_auroc": px_auroc, "pixel_auprc": px_auprc, "pro": pro,
             "map_res": MAP}
        results.append(r)
        print(f"{run}: img {img_auroc:.4f} | pixel AUROC {px_auroc:.4f} "
              f"AUPRC {px_auprc:.4f} | PRO {pro:.4f}  ({time.time() - t0:.0f}s)", flush=True)

    (OUT / "trackA.json").write_text(json.dumps(results, indent=2))
    print(f"\nsaved -> {OUT / 'trackA.json'}")


if __name__ == "__main__":
    main()
