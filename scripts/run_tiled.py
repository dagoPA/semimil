#!/usr/bin/env python
"""Tiled, multi-scale one-class detector: does finer token resolution recover small defects?

A DINOv2 ViT-S/14 at 518 px gives a 37x37 token grid on a 512 px MIIC image, so each
token covers about 14 px.  Defects below 0.1 % of the image (an equivalent side of
about 16 px) occupy one token or less, which is where the paper's detector loses
recall (0.636 at 1 % FPR).  Here each image is cut into a g x g grid of overlapping
tiles and every tile is upsampled to the backbone's input size, so tokens shrink to
about 8.6 px (g=2) or 5.8 px (g=3) without retraining anything.

Each *scale* (grid, overlap, input size) gets its own memory bank of normal tile
tokens.  Per-scale anomaly maps are stitched back to image coordinates (max over
overlapping tiles).  Scales are fused label-free: every scale's image score and map
are standardized with the mean and std of a fixed calibration set of *training*
normals that never enters a memory bank, and the fused score is the mean of those
z-scores.

Outputs, under ``output.dir``:
    raw/<scale>_s<seed>.npz   test + calibration image scores and 64x64 test maps
    results.json / .csv       per (method, seed) metrics, method = scale or fusion
    summary.txt               mean +- std over seeds, including per-size recall

Argument-free: reads ``configs/tiled.yaml``.  Re-running skips any scale/seed whose
raw file already exists and only recomputes the metrics.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_patchcore import build_extractor, list_images, load_image, min_dist_to_memory  # noqa: E402

from semimil.utils import seed_everything  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
# SEMIMIL_CONFIG lets a SLURM job point at another YAML (e.g. a smoke test).
CONFIG_PATH = Path(os.environ.get("SEMIMIL_CONFIG", REPO_ROOT / "configs" / "tiled.yaml"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NATIVE = 512  # MIIC images are 512 x 512


# ----------------------------------------------------------------------------- #
# tiling
# ----------------------------------------------------------------------------- #
def tile_boxes(grid: int, overlap: float, side: int = NATIVE) -> list[tuple[int, int, int]]:
    """(y0, x0, size) of a grid x grid set of square tiles covering a side x side image."""
    if grid == 1:
        return [(0, 0, side)]
    size = min(side, int(round(side / grid * (1.0 + overlap))))
    starts = np.linspace(0, side - size, grid).round().astype(int).tolist()
    return [(y, x, size) for y in starts for x in starts]


def tiles_of(img: torch.Tensor, boxes, input_size: int) -> torch.Tensor:
    """[3, 512, 512] normalized image -> [n_tiles, 3, input_size, input_size]."""
    crops = [img[:, y : y + s, x : x + s] for y, x, s in boxes]
    return torch.stack(
        [
            F.interpolate(c[None], size=(input_size, input_size), mode="bilinear",
                          align_corners=False)[0]
            for c in crops
        ]
    )


@torch.no_grad()
def tile_tokens(paths, extractor, boxes, input_size, bs):
    """Yield (path, [n_tiles, g, g, D] tokens) for each image, batching tiles on the GPU."""
    n_t = len(boxes)
    per_batch = max(1, bs // n_t)
    for i in range(0, len(paths), per_batch):
        chunk = paths[i : i + per_batch]
        x = torch.cat([tiles_of(load_image(p, NATIVE), boxes, input_size) for p in chunk])
        f = extractor(x)  # [len(chunk)*n_t, g, g, D]
        for j, p in enumerate(chunk):
            yield p, f[j * n_t : (j + 1) * n_t]


def stitch(dist: torch.Tensor, boxes, map_res: int) -> torch.Tensor:
    """[n_tiles, g, g] token distances -> [map_res, map_res] image map (max over overlaps)."""
    out = torch.full((map_res, map_res), -float("inf"), device=dist.device)
    k = map_res / NATIVE
    for t, (y, x, s) in enumerate(boxes):
        y0, x0 = int(round(y * k)), int(round(x * k))
        n = int(round(s * k))
        up = F.interpolate(dist[t][None, None], size=(n, n), mode="bilinear",
                           align_corners=False)[0, 0]
        ny, nx = min(n, map_res - y0), min(n, map_res - x0)  # rounding at the far edge
        out[y0 : y0 + ny, x0 : x0 + nx] = torch.maximum(
            out[y0 : y0 + ny, x0 : x0 + nx], up[:ny, :nx]
        )
    return torch.where(torch.isinf(out), torch.zeros_like(out), out)


# ----------------------------------------------------------------------------- #
# one scale, all seeds
# ----------------------------------------------------------------------------- #
def run_scale(sc, cfg, extractor, train_pool, calib, test_paths, seeds, raw_dir):
    boxes = tile_boxes(sc["grid"], sc.get("overlap", 0.0))
    size = sc["input_size"]
    bs = sc.get("batch_size", cfg["batch_size"])  # images x tiles per forward pass
    mem = cfg["memory"]
    coreset = sc.get("coreset_size", mem["coreset_size"])
    map_res = cfg["eval"]["map_res"]
    todo = [s for s in seeds if not (raw_dir / f"{sc['name']}_s{s}.npz").exists()]
    if not todo:
        print(f"[{sc['name']}] all seeds done, skipping", flush=True)
        return
    print(f"[{sc['name']}] grid={sc['grid']} tiles={len(boxes)} tile_px={boxes[0][2]} "
          f"input={size} token_px={boxes[0][2] / (size / 14):.1f} seeds={todo}", flush=True)

    # memory bank per seed
    banks = {}
    for s in todo:
        t0 = time.time()
        gen = torch.Generator().manual_seed(s)
        idx = torch.randperm(len(train_pool), generator=gen)[: mem["n_images"]]
        picks = []
        for _, f in tile_tokens([train_pool[j] for j in idx], extractor, boxes, size, bs):
            f = f.reshape(-1, f.shape[-1])  # all tiles' tokens of this image
            n_keep = mem["patches_per_tile"] * len(boxes)
            picks.append(f[torch.randint(0, f.shape[0], (n_keep,), generator=gen)].half().cpu())
        bank = torch.cat(picks)
        if bank.shape[0] > coreset:
            bank = bank[torch.randperm(bank.shape[0], generator=gen)[:coreset]]
        banks[s] = bank.to(DEVICE).float()
        print(f"  seed {s}: bank {tuple(bank.shape)} ({time.time() - t0:.0f}s)", flush=True)

    def score(paths, keep_maps):
        img = {s: [] for s in todo}
        maps = {s: [] for s in todo}
        for _, f in tile_tokens(paths, extractor, boxes, size, bs):
            g, d = f.shape[1], f.shape[-1]
            q = f.reshape(-1, d)
            for s in todo:
                dist = min_dist_to_memory(q, banks[s]).reshape(len(boxes), g, g)
                img[s].append(dist.max().item())
                if keep_maps:
                    m = stitch(dist, boxes, map_res)
                    maps[s].append(m.half().cpu().numpy())
        return img, maps

    t0 = time.time()
    c_img, c_map = score(calib, keep_maps=True)
    t_img, t_map = score(test_paths, keep_maps=True)
    print(f"  scored {len(calib)} calib + {len(test_paths)} test ({time.time() - t0:.0f}s)",
          flush=True)
    for s in todo:
        cm = np.stack(c_map[s]).astype(np.float32)
        np.savez_compressed(
            raw_dir / f"{sc['name']}_s{s}.npz",
            test_score=np.array(t_img[s]), calib_score=np.array(c_img[s]),
            test_map=np.stack(t_map[s]),
            calib_map_mean=np.float32(cm.mean()), calib_map_std=np.float32(cm.std()),
        )


# ----------------------------------------------------------------------------- #
# metrics
# ----------------------------------------------------------------------------- #
def size_bins(abn_paths, mask_dir):
    areas = []
    for p in abn_paths:
        m = np.asarray(Image.open(mask_dir / f"{p.stem}_mask.jpg").convert("L")) > 127
        areas.append(m.mean() * 100)
    return np.array(areas)


def masks_at(abn_paths, mask_dir, res):
    return np.stack([
        (np.asarray(Image.open(mask_dir / f"{p.stem}_mask.jpg").convert("L")
                    .resize((res, res), Image.NEAREST)) > 127).astype(np.uint8)
        for p in abn_paths
    ])


def metrics(score, amap, y, px_true, areas, bins, n_norm):
    normal = score[:n_norm]
    abn = score[n_norm:]
    thr = float(np.percentile(normal, 99))  # same operating point as the paper
    out = {
        "image_auroc": float(roc_auc_score(y, score)),
        "image_auprc": float(average_precision_score(y, score)),
        "recall_at_1fpr": float((abn >= thr).mean()),
        "pixel_auroc": float(roc_auc_score(px_true, amap.reshape(-1))),
    }
    for lo, hi, name in bins:
        sel = (areas >= lo) & (areas < hi)
        if not sel.any():  # only possible on a truncated (smoke) test set
            out[f"recall_{name}"] = out[f"auroc_{name}"] = float("nan")
            continue
        out[f"recall_{name}"] = float((abn[sel] >= thr).mean())
        yy = np.concatenate([np.zeros(n_norm), np.ones(sel.sum())])
        out[f"auroc_{name}"] = float(roc_auc_score(yy, np.concatenate([normal, abn[sel]])))
    return out


def evaluate(cfg, scales, seeds, raw_dir, out_dir, test_norm, test_abn, mask_dir):
    n_norm = len(test_norm)
    y = np.array([0] * n_norm + [1] * len(test_abn))
    areas = size_bins(test_abn, mask_dir)
    bins = [tuple(b) for b in cfg["eval"]["size_bins"]]
    res = cfg["eval"]["pixel_res"]
    px_true = np.concatenate([np.zeros((n_norm, res, res), np.uint8),
                              masks_at(test_abn, mask_dir, res)]).reshape(-1)

    def load(name, s):
        z = np.load(raw_dir / f"{name}_s{s}.npz")
        amap = torch.from_numpy(z["test_map"].astype(np.float32))[:, None]
        amap = F.interpolate(amap, size=(res, res), mode="bilinear",
                             align_corners=False)[:, 0].numpy()
        return z, amap

    methods = {sc["name"]: [sc["name"]] for sc in scales}
    methods.update({f"fuse:{'+'.join(f)}": f for f in cfg["fusions"]})
    rows = []
    for mname, parts in methods.items():
        for s in seeds:
            if not all((raw_dir / f"{p}_s{s}.npz").exists() for p in parts):
                continue
            zs, zm = [], []
            for p in parts:
                z, amap = load(p, s)
                if len(parts) == 1:
                    zs.append(z["test_score"])
                    zm.append(amap)
                else:  # label-free standardization on training-normal calibration set
                    c = z["calib_score"]
                    zs.append((z["test_score"] - c.mean()) / c.std())
                    zm.append((amap - z["calib_map_mean"]) / z["calib_map_std"])
            m = metrics(np.mean(zs, 0), np.mean(zm, 0), y, px_true, areas, bins, n_norm)
            rows.append({"method": mname, "seed": s, **m})
            print(f"{mname:<34} seed={s} AUROC={m['image_auroc']:.4f} "
                  f"AUPRC={m['image_auprc']:.4f} R@1%={m['recall_at_1fpr']:.3f} "
                  f"R_small={m[f'recall_{bins[0][2]}']:.3f}", flush=True)

    (out_dir / "results.json").write_text(json.dumps(rows, indent=2))
    with (out_dir / "results.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    by = defaultdict(list)
    for r in rows:
        by[r["method"]].append(r)
    keys = [k for k in rows[0] if k not in ("method", "seed")]
    counts = {name: int(((areas >= lo) & (areas < hi)).sum()) for lo, hi, name in bins}
    lines = [f"size bins (n abnormal images): {counts}", "mean +- std over seeds", ""]
    for mname, rs in by.items():
        cells = []
        for k in keys:
            v = np.array([r[k] for r in rs])
            cells.append(f"{k}={v.mean():.3f}+-{v.std():.3f}")
        lines.append(f"{mname} (n={len(rs)}): " + "  ".join(cells))
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    cfg = yaml.safe_load(CONFIG_PATH.open())
    seed_everything(cfg["seed"])
    raw = Path(cfg["data"]["raw_dir"])
    out_dir = Path(cfg["output"]["dir"])
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    train = list_images(sorted(raw.glob(cfg["data"]["train_normal_glob"])))
    test_norm = list_images(sorted(Path(raw, cfg["data"]["test_normal_dir"]).iterdir()))
    test_abn = list_images(sorted(Path(raw, cfg["data"]["test_abnormal_dir"]).iterdir()))
    mask_dir = Path(raw, cfg["data"]["test_mask_dir"])
    if cfg["limit_test"]:
        test_norm, test_abn = test_norm[: cfg["limit_test"]], test_abn[: cfg["limit_test"]]

    # fixed calibration set of training normals, disjoint from every memory bank
    gen = torch.Generator().manual_seed(cfg["seed"])
    perm = torch.randperm(len(train), generator=gen).tolist()
    n_cal = cfg["calibration"]["n_images"]
    calib = [train[j] for j in perm[:n_cal]]
    pool = [train[j] for j in perm[n_cal:]]
    seeds = cfg["seeds"]
    scales = [sc for sc in cfg["scales"] if sc.get("enabled", True)]
    print(f"device={DEVICE} train_pool={len(pool)} calib={len(calib)} "
          f"test={len(test_norm)}N/{len(test_abn)}A scales={[s['name'] for s in scales]}",
          flush=True)

    extractor = build_extractor(cfg["model"], DEVICE)
    for sc in scales:
        run_scale(sc, cfg, extractor, pool, calib, test_norm + test_abn, seeds, raw_dir)
    evaluate(cfg, scales, seeds, raw_dir, out_dir, test_norm, test_abn, mask_dir)


if __name__ == "__main__":
    main()
