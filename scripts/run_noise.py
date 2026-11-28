#!/usr/bin/env python
"""Robustness of the baseline and proposed detectors to image noise on MIIC.

A faster SEM recipe averages fewer frames and returns noisier images. We add zero-mean
Gaussian noise of standard deviation sigma (in gray levels of the 8-bit image) to each
512x512 image before resizing, rounded and clipped to [0, 255], with one fixed noise
realization per image and sigma, so every detector sees the same noisy images.
Two settings are scored for each sigma:
  - bank0:     memory bank from the clean normal images, noisy test and calibration
               images (the line changes its recipe and keeps the old memory bank);
  - bank<s>:   memory bank, test and calibration images all at the same sigma
               (the memory bank is rebuilt under the new recipe).

Runs (``configs/noise.yaml``):
  - exhaustive runs reproduce scripts/run_bigbank.py (every token of the 23,688 pool
    images, exact search). The memory bank is written straight to the GPU in fp16,
    centered on the mean of its first 2 M tokens, so host memory stays small. Writes
    raw/<name>_bank<b>_s<sigma>.npz with test_map and calib_map (37x37 distances).
  - sampled runs reproduce the baseline detector of scripts/run_tiled.py (seeds 0-2,
    1,000 images, 256 random tokens each, 100,000 tokens). Writes
    raw/<name>_bank<b>_s<sigma>_seed<k>.npz with test_score and test_map.
A file that exists is skipped. sigma = 0 is checked against the stored clean runs.

Argument-free: SEMIMIL_CONFIG selects the config (default configs/noise.yaml).

    CUDA_VISIBLE_DEVICES=1 NUMPY_MADVISE_HUGEPAGE=0 uv run python scripts/run_noise.py
"""

from __future__ import annotations

import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_bigbank  # noqa: E402
import run_patchcore  # noqa: E402
import run_tiled  # noqa: E402
from run_appendix_oneclass import bank_for, score  # noqa: E402
from run_patchcore import IMAGENET_MEAN, IMAGENET_STD, build_extractor, list_images  # noqa: E402

from semimil.utils import seed_everything  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(os.environ.get("SEMIMIL_CONFIG", REPO_ROOT / "configs" / "noise.yaml"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = True
SIGMA = {"value": 0.0}  # noise of the images loaded next


def noisy_load_image(path: Path, size: int) -> torch.Tensor:
    """run_patchcore.load_image with Gaussian noise of SIGMA gray levels at native size."""
    im = Image.open(path).convert("L")
    s = SIGMA["value"]
    if s > 0:
        a = np.asarray(im, dtype=np.float32)
        rng = np.random.default_rng([int(round(s * 100)), zlib.crc32(Path(path).name.encode())])
        a = np.clip(np.rint(a + rng.normal(0.0, s, a.shape)), 0, 255).astype(np.uint8)
        im = Image.fromarray(a)
    im = im.resize((size, size), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0)
    x = x.unsqueeze(0).repeat(3, 1, 1)
    return (x - IMAGENET_MEAN[0]) / IMAGENET_STD[0]


for module in (run_patchcore, run_bigbank, run_tiled):  # every loader the detectors use
    module.load_image = noisy_load_image


@torch.no_grad()
def gpu_bank(paths, extractor, size, bs):
    """Every token of paths, centered and in fp16 on the GPU, with fp32 squared norms."""
    first, n_first, mu, gb, pos = [], 0, None, None, 0
    for i in range(0, len(paths), bs):
        x = torch.stack([noisy_load_image(p, size) for p in paths[i : i + bs]])
        f = extractor(x)
        f = f.reshape(-1, f.shape[-1])
        if mu is None:
            first.append(f.half().cpu())
            n_first += len(f)
            if n_first < 2_000_000 and i + bs < len(paths):
                continue
            head = torch.cat(first)
            mu = head[:2_000_000].float().mean(0).to(DEVICE)
            per_img = len(head) // min(len(paths), i + bs)
            gb = torch.empty((len(paths) * per_img, head.shape[1]), dtype=torch.float16, device=DEVICE)
            for j in range(0, len(head), 500_000):
                blk = head[j : j + 500_000].to(DEVICE).float() - mu
                gb[pos : pos + len(blk)] = blk.half()
                pos += len(blk)
            del first, head
            continue
        gb[pos : pos + len(f)] = (f.float() - mu).half()
        pos += len(f)
    assert pos == len(gb), (pos, len(gb))
    b2 = torch.cat([(gb[i : i + 1_000_000].float() ** 2).sum(1) for i in range(0, len(gb), 1_000_000)])
    return gb, b2, mu


@torch.no_grad()
def nn_dist(queries, gb, b2, mu, q_chunk, b_chunk):
    """Exact nearest-neighbor distance of every query token, as run_bigbank.prefix_nn."""
    out = torch.empty(len(queries))
    for qs in range(0, len(queries), q_chunk):
        qf = queries[qs : qs + q_chunk].to(DEVICE).float() - mu
        q2 = (qf * qf).sum(1, keepdim=True)
        q = qf.half()
        best = torch.full((len(q),), float("inf"), device=DEVICE)
        for pos in range(0, len(gb), b_chunk):
            d2 = (q @ gb[pos : pos + b_chunk].T).float()
            d2.mul_(-2.0).add_(q2).add_(b2[pos : pos + b_chunk][None, :])
            best = torch.minimum(best, d2.min(1).values.clamp_min(0))
            del d2
        out[qs : qs + len(q)] = best.sqrt().cpu()
    return out


def splits(cfg):
    d, raw = cfg["data"], Path(cfg["data"]["raw_dir"])
    train = list_images(sorted(raw.glob(d["train_normal_glob"])))
    test = (list_images(sorted(Path(raw, d["test_normal_dir"]).iterdir()))
            + list_images(sorted(Path(raw, d["test_abnormal_dir"]).iterdir())))
    perm = torch.randperm(len(train), generator=torch.Generator().manual_seed(cfg["seed"])).tolist()
    n_cal = cfg["calibration"]["n_images"]
    return [train[j] for j in perm[:n_cal]], [train[j] for j in perm[n_cal:]], test


def exhaustive(run, cfg, calib, pool, test, out):
    order = torch.randperm(len(pool), generator=torch.Generator().manual_seed(run["bank_seed"])).tolist()
    pool = [pool[j] for j in order]
    extractor = build_extractor(run["model"], DEVICE)
    size, bs, n_t = run["input_size"], cfg["batch_size"], len(test)
    for bank_sigma in [0.0] + cfg["sigmas"]:
        todo = [s for s in cfg["sigmas"] if bank_sigma in (0.0, s)
                and not (out / f"{run['name']}_bank{bank_sigma:g}_s{s:g}.npz").exists()]
        check = bank_sigma == 0.0 and run.get("check_against")
        if not todo and not check:
            continue
        t0 = time.time()
        SIGMA["value"] = bank_sigma
        gb, b2, mu = gpu_bank(pool, extractor, size, bs)
        print(f"[{run['name']}] bank sigma {bank_sigma:g}: {tuple(gb.shape)} ({time.time() - t0:.0f}s)", flush=True)
        if check:  # clean queries against the stored clean run
            SIGMA["value"] = 0.0
            q = run_bigbank.extract(test[: cfg["check_images"]], extractor, size, bs)
            d = nn_dist(q, gb, b2, mu, cfg["q_chunk"], cfg["b_chunk"]).numpy()
            ref = np.load(run["check_against"])["test_map"][: cfg["check_images"]].astype(np.float32).ravel()
            print(f"[{run['name']}] check vs stored clean maps: max abs diff {np.abs(d - ref).max():.4f}, "
                  f"mean distance {ref.mean():.3f}", flush=True)
        for s in todo:
            t0 = time.time()
            SIGMA["value"] = s
            q = run_bigbank.extract(test + calib, extractor, size, bs)
            g = int(round((len(q) // (n_t + len(calib))) ** 0.5))
            d = nn_dist(q, gb, b2, mu, cfg["q_chunk"], cfg["b_chunk"]).reshape(-1, g, g)
            np.savez_compressed(out / f"{run['name']}_bank{bank_sigma:g}_s{s:g}.npz",
                                test_map=d[:n_t].half().numpy(), calib_map=d[n_t:].half().numpy())
            print(f"[{run['name']}] bank {bank_sigma:g}, images {s:g}: {time.time() - t0:.0f}s", flush=True)
        del gb, b2
        torch.cuda.empty_cache()


def sampled(run, cfg, pool, test, out):
    extractor = build_extractor(run["model"], DEVICE)
    boxes = run_tiled.tile_boxes(1, 0.0)
    size, bs = run["input_size"], cfg["batch_size"]
    for seed in run["seeds"]:
        for bank_sigma in [0.0] + cfg["sigmas"]:
            sig = [0.0] + cfg["sigmas"] if bank_sigma == 0.0 else [bank_sigma]
            todo = [s for s in sig if not (out / f"{run['name']}_bank{bank_sigma:g}_s{s:g}_seed{seed}.npz").exists()]
            if not todo:
                continue
            SIGMA["value"] = bank_sigma
            bank = bank_for(seed, pool, extractor, boxes, size, bs, cfg["memory"], cfg["memory"]["n_images"])
            for s in todo:
                SIGMA["value"] = s
                ts, tm = score(test, extractor, boxes, size, bs, bank, keep_maps=True)
                np.savez_compressed(out / f"{run['name']}_bank{bank_sigma:g}_s{s:g}_seed{seed}.npz",
                                    test_score=ts, test_map=tm)
                msg = f"[{run['name']}] seed {seed}, bank {bank_sigma:g}, images {s:g}"
                if s == 0.0 and run.get("check_against"):
                    ref = np.load(run["check_against"].format(seed))["test_score"][: len(ts)]
                    msg += f", Pearson r with the stored clean run {np.corrcoef(ts, ref)[0, 1]:.6f}"
                print(msg, flush=True)
            del bank
            torch.cuda.empty_cache()


def main() -> None:
    cfg = yaml.safe_load(CONFIG_PATH.open())
    seed_everything(cfg["seed"])
    out = Path(cfg["output"]["dir"]) / "raw"
    out.mkdir(parents=True, exist_ok=True)
    calib, pool, test = splits(cfg)
    if cfg.get("limit_test"):  # smoke test only
        test, pool = test[: cfg["limit_test"]] + test[-cfg["limit_test"]:], pool[: cfg["limit_pool"]]
    print(f"device={DEVICE} pool={len(pool)} calib={len(calib)} test={len(test)} sigmas={cfg['sigmas']}", flush=True)
    for run in cfg["runs"]:
        if run["name"] not in os.environ.get("RUNS", run["name"]).split(","):
            continue
        if run["bank"] == "exhaustive":
            exhaustive(run, cfg, calib, pool, test, out)
        else:
            sampled(run, cfg, pool, test, out)


if __name__ == "__main__":
    main()
