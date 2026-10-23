#!/usr/bin/env python
"""Structure-conditional (Mondrian) calibration of patch anomaly scores.

Diagnosis this answers: on MIIC the 1 % false-positive threshold is set by normal images
whose rare structures (line ends, vias, wide planes) have naturally large nearest-
neighbour distances, while a particle on a regular line raises the distance of a
structure whose distances are naturally tight.  A single global scale for all patches
therefore penalizes rare-but-normal structures and under-weights defects on common ones.
(PatchCore even *raises* the score of patches matched to rare memory entries.)

Here every patch is judged against patches of its own structure type:
1. structure types = k-means clusters of the frozen patch tokens of training normals;
2. per type, the nearest-neighbour distances of calibration-normal patches (never in the
   memory bank) give an empirical null distribution;
3. a test patch's score is its conformal p-value within its type,
   p = (1 + #{calibration distances of the type >= d}) / (n_type + 1),
   reported as -log10 p; the image score is the maximum over patches.
Mondrian conformal calibration controls the patch false-alarm rate within each type.

This script extracts tokens and assigns structure types (GPU); distances come from the
exhaustive-bank maps of run_bigbank.py.  Writes outputs/structcal/<rep>/labels_k<K>.npz.
Argument-free: configs/structcal.yaml (or SEMIMIL_CONFIG).
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
from run_patchcore import build_extractor, list_images, load_image  # noqa: E402

from semimil.utils import seed_everything  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(os.environ.get("SEMIMIL_CONFIG", REPO_ROOT / "configs" / "structcal.yaml"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def tokens(paths, extractor, size, bs):
    out = []
    for i in range(0, len(paths), bs):
        f = extractor(torch.stack([load_image(p, size) for p in paths[i : i + bs]]))
        out.append(f.reshape(f.shape[0], -1, f.shape[-1]).half().cpu())
    return torch.cat(out)  # [N, g*g, D]


@torch.no_grad()
def kmeans(x, k, iters, gen):
    """Plain Lloyd k-means on the GPU (cosine-free, Euclidean, as the detector)."""
    c = x[torch.randperm(len(x), generator=gen)[:k]].clone()
    for _ in range(iters):
        a = torch.cdist(x, c).argmin(1)
        for j in range(k):
            m = a == j
            if m.any():
                c[j] = x[m].mean(0)
    return c


@torch.no_grad()
def assign(feats, centers, chunk=65536):
    flat = feats.reshape(-1, feats.shape[-1])
    lab = torch.empty(len(flat), dtype=torch.int16)
    for i in range(0, len(flat), chunk):
        lab[i : i + chunk] = torch.cdist(flat[i : i + chunk].to(DEVICE).float(), centers).argmin(1).cpu()
    return lab.reshape(feats.shape[:2])


def main() -> None:
    cfg = yaml.safe_load(CONFIG_PATH.open())
    seed_everything(cfg["seed"])
    raw = Path(cfg["data"]["raw_dir"])
    train = list_images(sorted(raw.glob(cfg["data"]["train_normal_glob"])))
    test = (list_images(sorted(Path(raw, cfg["data"]["test_normal_dir"]).iterdir()))
            + list_images(sorted(Path(raw, cfg["data"]["test_abnormal_dir"]).iterdir())))
    gen = torch.Generator().manual_seed(cfg["seed"])  # same calibration split as run_bigbank
    perm = torch.randperm(len(train), generator=gen).tolist()
    n_cal = cfg["calibration"]["n_images"]
    calib, pool = [train[j] for j in perm[:n_cal]], [train[j] for j in perm[n_cal:]]
    g2 = torch.Generator().manual_seed(cfg["kmeans"]["seed"])
    fit_imgs = [pool[j] for j in torch.randperm(len(pool), generator=g2)[: cfg["kmeans"]["n_images"]].tolist()]

    for rep in cfg["representations"]:
        out = Path(cfg["output"]["dir"]) / rep["name"]
        out.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        extractor = build_extractor(rep["model"], DEVICE)
        size, bs = cfg["input_size"], cfg["batch_size"]
        fit = tokens(fit_imgs, extractor, size, bs)
        fit = fit.reshape(-1, fit.shape[-1])
        fit = fit[torch.randperm(len(fit), generator=g2)[: cfg["kmeans"]["n_tokens"]]].to(DEVICE).float()
        cal_f = tokens(calib, extractor, size, bs)
        test_f = tokens(test, extractor, size, bs)
        print(f"[{rep['name']}] tokens: fit {tuple(fit.shape)} calib {tuple(cal_f.shape)} test "
              f"{tuple(test_f.shape)} ({time.time() - t0:.0f}s)", flush=True)
        for k in cfg["kmeans"]["k"]:
            centers = kmeans(fit, k, cfg["kmeans"]["iters"], g2)
            np.savez_compressed(out / f"labels_k{k}.npz",
                                calib=assign(cal_f, centers).numpy(), test=assign(test_f, centers).numpy(),
                                centers=centers.cpu().numpy())
            print(f"  k={k} done ({time.time() - t0:.0f}s)", flush=True)
        del extractor
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
