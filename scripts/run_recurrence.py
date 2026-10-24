#!/usr/bin/env python
"""Recurrence scoring: distance to the k-th nearest *distinct normal image* in the exhaustive memory.

With every training token in memory, a rare but normal structure recurs in a few images, while a
particle is unique.  The nearest-neighbor distance cannot tell the two apart once the structure is
covered; the distance to the k-th nearest distinct image can, in principle: a recurring structure
has k close images, a defect has none.  (Neighbors are counted per image, so the overlapping
patches of one image never count as recurrence.)

For one token representation, every test and calibration token gets its NN distance to each of
the 23,688 pool images (tokens of one image are contiguous, so a block's distances are reduced per
image with one min); the k smallest per-image distances are kept.  Saves raw/k<k>.npz
(test_map, calib_map) for every k in ``ks``, in the format of run_bigbank.py; k = 1 reproduces
raw/n23688.npz of run_bigbank.py.  Argument-free: configs/recurrence_<rep>.yaml.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_bigbank import extract  # noqa: E402
from run_patchcore import build_extractor, list_images  # noqa: E402

from semimil.utils import seed_everything  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(os.environ.get("SEMIMIL_CONFIG", REPO_ROOT / "configs" / "recurrence_L8.yaml"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def knn_images(queries, bank, per_img, kmax, q_chunk, imgs_per_block):
    """[Q, kmax] sorted distances of every query to its kmax nearest distinct bank images."""
    mu = bank[: min(len(bank), 2_000_000)].float().mean(0)
    gb = torch.empty(bank.shape, dtype=torch.float16, device=DEVICE)
    for i in range(0, len(bank), 1_000_000):
        gb[i : i + 1_000_000] = (bank[i : i + 1_000_000].float() - mu).half().to(DEVICE)
    b2 = torch.cat([(gb[i : i + 1_000_000].float() ** 2).sum(1) for i in range(0, len(gb), 1_000_000)])
    mu = mu.to(DEVICE)
    n_img = len(bank) // per_img
    blk = imgs_per_block * per_img
    out = torch.empty((len(queries), kmax))
    for qs in range(0, len(queries), q_chunk):
        qf = queries[qs : qs + q_chunk].to(DEVICE).float() - mu
        q2 = (qf * qf).sum(1, keepdim=True)
        q = qf.half()
        best = torch.full((len(q), kmax), float("inf"), device=DEVICE)
        for pos in range(0, n_img * per_img, blk):
            end = min(pos + blk, n_img * per_img)
            d2 = (q @ gb[pos:end].T).float()
            d2.mul_(-2.0).add_(q2).add_(b2[None, pos:end])
            per = torch.amin(d2.view(len(q), -1, per_img), 2)  # [q, images in block]
            del d2
            best = torch.topk(torch.cat([best, per], 1), kmax, dim=1, largest=False).values
        out[qs : qs + len(q)] = best.clamp_min_(0).sqrt_().cpu()
    del gb, b2
    torch.cuda.empty_cache()
    return out


def main() -> None:
    cfg = yaml.safe_load(CONFIG_PATH.open())
    seed_everything(cfg["seed"])
    raw = Path(cfg["data"]["raw_dir"])
    out_dir = Path(cfg["output"]["dir"])
    (out_dir / "raw").mkdir(parents=True, exist_ok=True)
    train = list_images(sorted(raw.glob(cfg["data"]["train_normal_glob"])))
    test_norm = list_images(sorted(Path(raw, cfg["data"]["test_normal_dir"]).iterdir()))
    test_abn = list_images(sorted(Path(raw, cfg["data"]["test_abnormal_dir"]).iterdir()))
    if cfg.get("limit_test"):
        test_norm, test_abn = test_norm[: cfg["limit_test"]], test_abn[: cfg["limit_test"]]
    gen = torch.Generator().manual_seed(cfg["seed"])  # same calibration split as run_bigbank
    perm = torch.randperm(len(train), generator=gen).tolist()
    n_cal = cfg["calibration"]["n_images"]
    calib, pool = [train[j] for j in perm[:n_cal]], [train[j] for j in perm[n_cal:]]
    if cfg.get("pool_images"):
        pool = pool[: cfg["pool_images"]]
    size, bs = cfg["input_size"], cfg["batch_size"]
    print(f"device={DEVICE} pool={len(pool)} calib={len(calib)} test={len(test_norm)}N/{len(test_abn)}A", flush=True)
    extractor = build_extractor(cfg["model"], DEVICE)
    t0 = time.time()
    bank = extract(pool, extractor, size, bs)
    q = torch.cat([extract(test_norm + test_abn, extractor, size, bs), extract(calib, extractor, size, bs)])
    del extractor
    torch.cuda.empty_cache()
    per_img = bank.shape[0] // len(pool)
    print(f"tokens: bank {tuple(bank.shape)}, queries {tuple(q.shape)} ({time.time() - t0:.0f}s)", flush=True)
    t0 = time.time()
    ks = cfg["ks"]
    d = knn_images(q, bank, per_img, max(ks), cfg["q_chunk"], cfg["imgs_per_block"])
    print(f"per-image kNN, k <= {max(ks)} ({time.time() - t0:.0f}s)", flush=True)
    g = int(round(per_img ** 0.5))
    n_t = len(test_norm) + len(test_abn)
    for k in ks:
        m = d[:, k - 1].reshape(-1, g, g)
        np.savez_compressed(out_dir / "raw" / f"k{k}.npz", test_map=m[:n_t].half().numpy(),
                            calib_map=m[n_t:].half().numpy())
        s = m[:n_t].reshape(n_t, -1).max(1).values.numpy()
        nn_, sn, sa = len(test_norm), s[: len(test_norm)], s[len(test_norm):]
        thr = np.percentile(sn, 99)
        print(f"k={k}: Sens@99 {(sa >= thr).mean():.3f}  thr {thr:.3f}", flush=True)


if __name__ == "__main__":
    main()
