#!/usr/bin/env python
"""One-class detectors of Appendix B under the protocol of the main text.

Each run builds its memory banks exactly as scripts/run_tiled.py builds the baseline
detector (one tile per image, seeds 0-2, 1,000 images from the training normals outside
the 200 calibration images, 256 random tokens per image, then 100,000 tokens at random),
so the DINOv2 run reproduces the main-text baseline and the other extractors and the
ablations differ from it only in the factor they change.

For every run and seed it writes raw/<name>_s<seed>.npz with the image scores of the
real MIIC test (normal then defective, sorted by name), the per-token nearest-neighbor
distance maps of that test, and, when the run asks for it, the image scores of the
synthetic test (Inpainting_test, originals then anomalies). A run whose file exists is
skipped. The DINOv2 run is checked against the stored main-text scores, which it
matches up to the floating-point nondeterminism of the GPU.

The runs listed under ``timing`` are then timed once more on the first seed, alone on
the GPU after a warm-up batch: building the memory bank and scoring the real test set.
The times go to timing.json, which is written once.

Argument-free: reads configs/appendix_oneclass.yaml.

    CUDA_VISIBLE_DEVICES=1 uv run python scripts/run_appendix_oneclass.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_patchcore import build_extractor, list_images, min_dist_to_memory
from run_tiled import tile_boxes, tile_tokens

from semimil.utils import seed_everything

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(os.environ.get("SEMIMIL_CONFIG",
                                  REPO_ROOT / "configs" / "appendix_oneclass.yaml"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAIN_BASELINE = "/autofs/space/crater_001/projects/micropath/semi/outputs/tiled/raw/g1_518_s{}.npz"


@torch.no_grad()
def bank_for(seed, pool, extractor, boxes, size, bs, mem, n_images):
    """Memory bank of one seed, drawn exactly as run_tiled.run_scale draws it."""
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(pool), generator=gen)[:n_images]
    picks = []
    for _, f in tile_tokens([pool[j] for j in idx], extractor, boxes, size, bs):
        f = f.reshape(-1, f.shape[-1])
        n_keep = mem["patches_per_image"] * len(boxes)
        picks.append(f[torch.randint(0, f.shape[0], (n_keep,), generator=gen)].half().cpu())
    bank = torch.cat(picks)
    if bank.shape[0] > mem["coreset_size"]:
        bank = bank[torch.randperm(bank.shape[0], generator=gen)[: mem["coreset_size"]]]
    return bank.to(DEVICE).float()


def sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def score(paths, extractor, boxes, size, bs, bank, keep_maps):
    scores, maps = [], []
    for _, f in tile_tokens(paths, extractor, boxes, size, bs):
        g, d = f.shape[1], f.shape[-1]
        dist = min_dist_to_memory(f.reshape(-1, d), bank).reshape(g, g)
        scores.append(dist.max().item())
        if keep_maps:
            maps.append(dist.half().cpu().numpy())
    return np.array(scores), (np.stack(maps) if keep_maps else None)


def main() -> None:
    cfg = yaml.safe_load(CONFIG_PATH.open())
    seed_everything(cfg["seed"])
    d = cfg["data"]
    raw = Path(d["raw_dir"])
    out = Path(cfg["output"]["dir"]) / "raw"
    out.mkdir(parents=True, exist_ok=True)

    train = list_images(sorted(raw.glob(d["train_normal_glob"])))
    gen = torch.Generator().manual_seed(cfg["seed"])  # calibration set, as in run_tiled.py
    perm = torch.randperm(len(train), generator=gen).tolist()
    pool = [train[j] for j in perm[cfg["calibration"]["n_images"]:]]
    test_norm = list_images(sorted(Path(raw, d["test_normal_dir"]).iterdir()))
    test_abn = list_images(sorted(Path(raw, d["test_abnormal_dir"]).iterdir()))
    test, y = test_norm + test_abn, np.r_[np.zeros(len(test_norm)), np.ones(len(test_abn))]
    syn = (list_images(sorted(raw.glob(d["syn_test_neg_glob"])))
           + list_images(sorted(raw.glob(d["syn_test_pos_glob"]))))
    boxes = tile_boxes(1, 0.0)
    print(f"device={DEVICE} pool={len(pool)} test={len(test)} synthetic={len(syn)}", flush=True)

    for run in cfg["runs"]:
        todo = [s for s in cfg["seeds"] if not (out / f"{run['name']}_s{s}.npz").exists()]
        if not todo:
            print(f"[{run['name']}] done, skipping", flush=True)
            continue
        extractor = build_extractor(run["model"], DEVICE)
        size, n_images = run["input_size"], run.get("n_images", cfg["memory"]["n_images"])
        bs = run.get("batch_size", cfg["batch_size"])
        for s in todo:
            t0 = time.time()
            bank = bank_for(s, pool, extractor, boxes, size, bs, cfg["memory"], n_images)
            ts, tm = score(test, extractor, boxes, size, bs, bank, keep_maps=True)
            payload = {"test_score": ts, "test_map": tm}
            if run.get("synthetic"):
                payload["syn_score"], _ = score(syn, extractor, boxes, size, bs, bank, False)
            np.savez_compressed(out / f"{run['name']}_s{s}.npz", **payload)
            msg = f"[{run['name']}] seed {s}: bank {tuple(bank.shape)}, {time.time() - t0:.0f}s"
            if run["name"] == "dinov2":
                ref = np.load(MAIN_BASELINE.format(s))["test_score"]
                msg += (f", AUROC {roc_auc_score(y, ts):.4f} "
                        f"(main text {roc_auc_score(y, ref):.4f}), "
                        f"Pearson r {np.corrcoef(ts, ref)[0, 1]:.6f}")
            print(msg, flush=True)
            del bank
            torch.cuda.empty_cache()
        del extractor

    timing = cfg.get("timing")
    path = Path(cfg["output"]["dir"]) / "timing.json"
    if timing and not path.exists():
        result = {"gpu": torch.cuda.get_device_name() if DEVICE == "cuda" else "cpu",
                  "seed": timing["seed"], "n_memory_images": cfg["memory"]["n_images"],
                  "n_test_images": len(test), "runs": {}}
        for run in (r for r in cfg["runs"] if r["name"] in timing["runs"]):
            extractor = build_extractor(run["model"], DEVICE)
            size, bs = run["input_size"], run.get("batch_size", cfg["batch_size"])
            with torch.no_grad():  # warm-up batch
                next(iter(tile_tokens(test[:bs], extractor, boxes, size, bs)))
            sync()
            t0 = time.perf_counter()
            bank = bank_for(timing["seed"], pool, extractor, boxes, size, bs, cfg["memory"],
                            run.get("n_images", cfg["memory"]["n_images"]))
            sync()
            t1 = time.perf_counter()
            score(test, extractor, boxes, size, bs, bank, keep_maps=False)
            sync()
            t2 = time.perf_counter()
            result["runs"][run["name"]] = {"memory_bank_s": t1 - t0, "test_s": t2 - t1,
                                           "total_s": t2 - t0}
            print(f"[timing] {run['name']}: bank {t1 - t0:.0f}s, test {t2 - t1:.0f}s", flush=True)
            del bank, extractor
            torch.cuda.empty_cache()
        path.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
