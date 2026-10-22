#!/usr/bin/env python
"""Periodic self-reference: compare every patch with its own repeated cells, in DINOv2 space.

IC layouts repeat.  On MIIC most images are periodic along one axis with a period of
about 7.3 px (image autocorrelation ~0.99), and some have a 2-D lattice.  A memory bank
compares a patch with *generic* normal patches from other images; a small defect only
shifts a 14 px token a little, so its nearest bank neighbor can be closer than the
natural spread of normal tokens.  Industrial inspection instead compares each cell with
its repeats (cell-to-cell / die-to-die), but in pixel space, with a known period, and it
is sensitive to process variation.  Here the comparison is made in foundation-model
feature space, with the period estimated from each image, and pixel-aligned:

1. Lattice vectors.  From the image's own autocorrelation, keep up to ``max_vectors``
   peaks with correlation >= ``min_corr`` and length in [``min_shift``, ``max_shift``]
   px (``min_shift`` keeps a small defect from overlapping its twin).
2. Aligned twins.  For each vector v, run the frozen backbone on the image translated by
   +v and by -v (reflect padding).  Token p of the translated image then covers exactly
   the content one lattice step away from token p of the original, so the two tokens
   see the same layout phase, with no sub-token misalignment.
3. Self score.  d(p) = distance between token p and its twin token, taken over every
   twin whose source window lies inside the image; the patch score is the mean of the
   ``k_twins`` smallest distances (a normal patch matches at least one twin, a defect
   matches none).  Image score = max over patches.
4. Fusion with the memory bank.  Bank and self scores are standardized with a fixed
   calibration set of training normals (never in the bank) and combined; images with
   no lattice vector fall back to the bank score alone.  No test label is used anywhere.

Outputs under ``output.dir``: raw/s<seed>.npz, results.json/.csv, summary.txt.
Argument-free: reads ``configs/selfref.yaml`` (or ``SEMIMIL_CONFIG``).
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
from localization import pro_auc  # noqa: E402
from run_patchcore import IMAGENET_MEAN, IMAGENET_STD, build_extractor, list_images, min_dist_to_memory  # noqa: E402

from semimil.utils import seed_everything  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(os.environ.get("SEMIMIL_CONFIG", REPO_ROOT / "configs" / "selfref.yaml"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------------- #
# lattice estimation and translated copies
# ----------------------------------------------------------------------------- #
def gray(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def lattice_vectors(img: np.ndarray, lc: dict) -> list[tuple[int, int]]:
    """Up to max_vectors (dy, dx) autocorrelation peaks, one per +-v pair."""
    a = img - img.mean()
    spec = np.fft.fft2(a)
    ac = np.real(np.fft.ifft2(spec * np.conj(spec)))
    if ac[0, 0] <= 0:
        return []
    ac = np.fft.fftshift(ac / ac[0, 0])
    h, w = ac.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[-cy : h - cy, -cx : w - cx]
    r = np.hypot(yy, xx)
    m = np.where((r >= lc["min_shift"]) & (r <= lc["max_shift"]), ac, -np.inf)
    out, nms = [], lc["nms_px"]
    while len(out) < lc["max_vectors"]:
        i = np.unravel_index(np.argmax(m), m.shape)
        if not np.isfinite(m[i]) or m[i] < lc["min_corr"]:
            break
        dy, dx = int(i[0] - cy), int(i[1] - cx)
        out.append((dy, dx))
        for sy, sx in ((i[0], i[1]), (2 * cy - i[0], 2 * cx - i[1])):  # the peak and its mirror
            m[max(0, sy - nms) : sy + nms + 1, max(0, sx - nms) : sx + nms + 1] = -np.inf
    return out


def translate(img: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """out[y, x] = img[y + dy, x + dx], reflect-padded outside the image."""
    p = max(abs(dy), abs(dx))
    big = np.pad(img, p, mode="reflect")
    h, w = img.shape
    return big[p + dy : p + dy + h, p + dx : p + dx + w]


def to_input(img: np.ndarray, size: int) -> torch.Tensor:
    x = torch.from_numpy(np.ascontiguousarray(img))[None, None]
    x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)[0]
    x = x.repeat(3, 1, 1)
    return (x - IMAGENET_MEAN[0]) / IMAGENET_STD[0]


def twin_valid(dy: int, dx: int, g: int, side: int) -> torch.Tensor:
    """[g, g] True where token p's twin window (shifted by +v) lies inside the image."""
    edges = np.linspace(0, side, g + 1)
    y0, y1 = edges[:-1] + dy, edges[1:] + dy
    x0, x1 = edges[:-1] + dx, edges[1:] + dx
    vy = (y0 >= 0) & (y1 <= side)
    vx = (x0 >= 0) & (x1 <= side)
    return torch.from_numpy(vy[:, None] & vx[None, :])


# ----------------------------------------------------------------------------- #
# per-image scoring
# ----------------------------------------------------------------------------- #
@torch.no_grad()
def score_image(path, extractor, banks, cfg):
    """Bank and self-reference token maps for one image.

    Returns (bank_maps {seed: [g,g]}, self_map [g,g] or None, n_vectors).
    """
    size = cfg["input_size"]
    img = gray(path)
    side = img.shape[0]
    vecs = lattice_vectors(img, cfg["lattice"])
    shifts = [s for v in vecs for s in (v, (-v[0], -v[1]))]
    x = torch.stack([to_input(img, size)] + [to_input(translate(img, *s), size) for s in shifts])
    f = extractor(x)  # [1 + n_shifts, g, g, D]
    g, d = f.shape[1], f.shape[-1]
    base = f[0]
    bank_maps = {s: min_dist_to_memory(base.reshape(-1, d), m).reshape(g, g)
                 for s, m in banks.items()}
    if not shifts:
        return bank_maps, None, 0
    dists = []
    for k, s in enumerate(shifts):
        dd = torch.linalg.vector_norm(base - f[k + 1], dim=-1)  # [g, g]
        dd[~twin_valid(*s, g, side).to(dd.device)] = float("inf")
        dists.append(dd)
    dist = torch.stack(dists)  # [n_shifts, g, g]
    kk = min(cfg["k_twins"], dist.shape[0])
    low = torch.topk(dist, kk, dim=0, largest=False).values
    finite = torch.isfinite(low)
    self_map = torch.where(finite, low, torch.zeros_like(low)).sum(0) / finite.sum(0).clamp(min=1)
    self_map[~finite.any(0)] = 0.0  # no valid twin at all (not reached with +-v pairs)
    return bank_maps, self_map, len(vecs)


def run(cfg, extractor, train, calib, test_paths, seeds, raw_dir):
    todo = [s for s in seeds if not (raw_dir / f"s{s}.npz").exists()]
    if not todo:
        return
    mem, size, bs = cfg["memory"], cfg["input_size"], cfg["batch_size"]
    banks = {}
    for s in todo:
        t0 = time.time()
        gen = torch.Generator().manual_seed(s)
        idx = torch.randperm(len(train), generator=gen)[: mem["n_images"]]
        paths = [train[j] for j in idx]
        picks = []
        for i in range(0, len(paths), bs):
            x = torch.stack([to_input(gray(p), size) for p in paths[i : i + bs]])
            f = extractor(x)
            f = f.reshape(f.shape[0], -1, f.shape[-1])
            for b in range(f.shape[0]):
                sel = torch.randint(0, f.shape[1], (mem["patches_per_image"],), generator=gen)
                picks.append(f[b, sel].half().cpu())
        bank = torch.cat(picks)
        if bank.shape[0] > mem["coreset_size"]:
            bank = bank[torch.randperm(bank.shape[0], generator=gen)[: mem["coreset_size"]]]
        banks[s] = bank.to(DEVICE).float()
        print(f"seed {s}: bank {tuple(bank.shape)} ({time.time() - t0:.0f}s)", flush=True)

    rec = {name: {"bank": defaultdict(list), "self": [], "nvec": []} for name in ("calib", "test")}
    t0 = time.time()
    for name, paths in (("calib", calib), ("test", test_paths)):
        for n, p in enumerate(paths):
            bmaps, smap, nv = score_image(p, extractor, banks, cfg)
            for s, bm in bmaps.items():
                rec[name]["bank"][s].append(bm.half().cpu().numpy())
            g = next(iter(bmaps.values())).shape[0]
            rec[name]["self"].append((smap if smap is not None else torch.zeros(g, g)).half().cpu().numpy())
            rec[name]["nvec"].append(nv)
            if (n + 1) % 200 == 0:
                print(f"  {name} {n + 1}/{len(paths)} ({time.time() - t0:.0f}s)", flush=True)
    for s in todo:
        np.savez_compressed(
            raw_dir / f"s{s}.npz",
            test_bank=np.stack(rec["test"]["bank"][s]), test_self=np.stack(rec["test"]["self"]),
            test_nvec=np.array(rec["test"]["nvec"]),
            calib_bank=np.stack(rec["calib"]["bank"][s]), calib_self=np.stack(rec["calib"]["self"]),
            calib_nvec=np.array(rec["calib"]["nvec"]),
        )


# ----------------------------------------------------------------------------- #
# metrics
# ----------------------------------------------------------------------------- #
def evaluate(cfg, seeds, raw_dir, out_dir, test_norm, test_abn, mask_dir):
    n_norm = len(test_norm)
    y = np.array([0] * n_norm + [1] * len(test_abn))
    res = cfg["eval"]["pixel_res"]
    areas = np.array([(np.asarray(Image.open(mask_dir / f"{p.stem}_mask.jpg").convert("L")) > 127).mean() * 100
                      for p in test_abn])
    masks = np.concatenate([
        np.zeros((n_norm, res, res), np.uint8),
        np.stack([(np.asarray(Image.open(mask_dir / f"{p.stem}_mask.jpg").convert("L")
                              .resize((res, res), Image.NEAREST)) > 127).astype(np.uint8) for p in test_abn]),
    ])
    bins = [tuple(b) for b in cfg["eval"]["size_bins"]]

    def up(maps):
        t = torch.from_numpy(maps.astype(np.float32))[:, None]
        return F.interpolate(t, size=(res, res), mode="bilinear", align_corners=False)[:, 0].numpy()

    rows = []
    for s in seeds:
        z = np.load(raw_dir / f"s{s}.npz")
        tb, ts, cb, cs = z["test_bank"].astype(np.float32), z["test_self"].astype(np.float32), \
            z["calib_bank"].astype(np.float32), z["calib_self"].astype(np.float32)
        t_has, c_has = z["test_nvec"] > 0, z["calib_nvec"] > 0
        # image scores (max over tokens) and label-free standardization on calibration normals
        bi, si = tb.reshape(len(tb), -1).max(1), ts.reshape(len(ts), -1).max(1)
        cbi, csi = cb.reshape(len(cb), -1).max(1), cs.reshape(len(cs), -1).max(1)
        zb = (bi - cbi.mean()) / cbi.std()
        zs = (si - csi[c_has].mean()) / csi[c_has].std()
        mb = (tb - cb.mean()) / cb.std()
        ms = (ts - cs[c_has].mean()) / cs[c_has].std()
        variants = {
            "bank": (bi, tb),
            "self": (np.where(t_has, zs, zb), np.where(t_has[:, None, None], ms, mb)),
            "mean": (np.where(t_has, (zb + zs) / 2, zb), np.where(t_has[:, None, None], (mb + ms) / 2, mb)),
            "max": (np.where(t_has, np.maximum(zb, zs), zb), np.where(t_has[:, None, None], np.maximum(mb, ms), mb)),
        }
        for name, (score, amap) in variants.items():
            amap = up(amap)
            normal, abn = score[:n_norm], score[n_norm:]
            thr = float(np.percentile(normal, 99))
            r = {"method": name, "seed": s,
                 "image_auroc": float(roc_auc_score(y, score)),
                 "image_auprc": float(average_precision_score(y, score)),
                 "recall_at_1fpr": float((abn >= thr).mean()),
                 "pixel_auroc": float(roc_auc_score(masks.reshape(-1), amap.reshape(-1))),
                 "pro": pro_auc(amap, masks)}
            for lo, hi, bn in bins:
                sel = (areas >= lo) & (areas < hi)
                r[f"recall_{bn}"] = float((abn[sel] >= thr).mean())
                yy = np.r_[np.zeros(n_norm), np.ones(sel.sum())]
                r[f"auroc_{bn}"] = float(roc_auc_score(yy, np.r_[normal, abn[sel]]))
            rows.append(r)
            print(f"{name:<5} seed={s} AUROC={r['image_auroc']:.4f} AUPRC={r['image_auprc']:.4f} "
                  f"R@1%={r['recall_at_1fpr']:.3f} R_small={r['recall_small']:.3f} "
                  f"pxAUROC={r['pixel_auroc']:.4f} PRO={r['pro']:.3f}", flush=True)
        print(f"  periodic images: test {t_has.mean():.3f}, calib {c_has.mean():.3f}", flush=True)

    (out_dir / "results.json").write_text(json.dumps(rows, indent=2))
    with (out_dir / "results.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    keys = [k for k in rows[0] if k not in ("method", "seed")]
    lines = ["mean +- std over seeds"]
    for name in dict.fromkeys(r["method"] for r in rows):
        rs = [r for r in rows if r["method"] == name]
        lines.append(f"{name} (n={len(rs)}): " + "  ".join(
            f"{k}={np.mean([r[k] for r in rs]):.3f}+-{np.std([r[k] for r in rs]):.3f}" for k in keys))
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
    # same calibration set as run_tiled.py (seed 42, first 200 of a permutation of train)
    gen = torch.Generator().manual_seed(cfg["seed"])
    perm = torch.randperm(len(train), generator=gen).tolist()
    n_cal = cfg["calibration"]["n_images"]
    calib, pool = [train[j] for j in perm[:n_cal]], [train[j] for j in perm[n_cal:]]
    print(f"device={DEVICE} pool={len(pool)} calib={len(calib)} test={len(test_norm)}N/{len(test_abn)}A",
          flush=True)
    extractor = build_extractor(cfg["model"], DEVICE)
    run(cfg, extractor, pool, calib, test_norm + test_abn, cfg["seeds"], raw_dir)
    evaluate(cfg, cfg["seeds"], raw_dir, out_dir, test_norm, test_abn, mask_dir)


if __name__ == "__main__":
    main()
