#!/usr/bin/env python
"""One-class detector on the external semiconductor datasets (Carinthia-S, Texture-AD Wafer).

Same detector as on MIIC: patch tokens of a (frozen or patch-adapted) DINOv2, a memory
bank of normal tokens, nearest-neighbor distance as the patch score, max over patches
as the image score, and optional tiling plus label-free fusion of scales (see
``run_tiled.py``).  Nothing is tuned on these datasets.

Evaluation units ("tasks") come from the manifests written by ``prepare_external.py``:
    carinthia  one task per fold: bank and calibration from the other folds' normals,
               test = the fold's held-out normals + every defect image
    texturead  one task per wafer subclass and photo fold: bank and calibration from
               the other folds' normal crops, test = every crop of the fold's photographs

Metrics per task: image AUROC/AUPRC, pixel AUROC and PRO (integrated to 30 % FPR) over
the test images, and for Carinthia the image AUROC of each defect class against the
held-out normals.  The Carinthia normals have a smoother background than the most
common defect class, so pixel AUROC and PRO are its primary metrics.

Outputs under ``output.dir``: raw/<dataset>/<extractor>__<scale>/<task>_s<seed>.npz,
results.json/.csv (per method, task, seed) and summary.txt.  Argument-free: reads
``configs/external.yaml`` (or ``SEMIMIL_CONFIG``); existing raw files are reused.
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
from run_patchcore import IMAGENET_MEAN, IMAGENET_STD, build_extractor, min_dist_to_memory  # noqa: E402
from run_tiled import NATIVE, stitch, tile_boxes, tiles_of  # noqa: E402

from semimil.utils import seed_everything  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(os.environ.get("SEMIMIL_CONFIG", REPO_ROOT / "configs" / "external.yaml"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------------- #
# data
# ----------------------------------------------------------------------------- #
RGB = False  # set per dataset in main(): MVTec is color, the semiconductor datasets are gray


def load(path: str, crop: int) -> torch.Tensor:
    """Image -> border-cropped, 512x512, ImageNet-normalized [3, 512, 512] (gray or RGB)."""
    im = Image.open(path).convert("RGB" if RGB else "L")
    if crop:
        im = im.crop((crop, crop, im.width - crop, im.height - crop))
    im = im.resize((NATIVE, NATIVE), Image.BILINEAR)
    a = np.asarray(im, dtype=np.float32) / 255.0
    x = torch.from_numpy(a).permute(2, 0, 1) if RGB else torch.from_numpy(a)[None].repeat(3, 1, 1)
    return (x - IMAGENET_MEAN[0]) / IMAGENET_STD[0]


def load_mask(path: str, crop: int, res: int) -> np.ndarray:
    if not path:
        return np.zeros((res, res), np.uint8)
    m = Image.open(path).convert("L")
    if crop:
        m = m.crop((crop, crop, m.width - crop, m.height - crop))
    return (np.asarray(m.resize((res, res), Image.NEAREST)) > 127).astype(np.uint8)


def read_manifest(path: Path) -> list[dict]:
    rows = list(csv.DictReader(path.open()))
    for r in rows:
        r["label"] = int(r["label"])
        if "fold" in r:
            r["fold"] = int(r["fold"])
    return rows


def make_tasks(name: str, rows: list[dict], cfg: dict) -> dict[str, dict]:
    """task id -> {"pool": normal indices for bank+calibration, "test": indices}."""
    tasks = {}
    if name == "carinthia":
        normals = [i for i, r in enumerate(rows) if r["label"] == 0]
        defects = [i for i, r in enumerate(rows) if r["label"] == 1]
        for k in sorted({rows[i]["fold"] for i in normals}):
            tasks[f"fold{k}"] = {
                "pool": [i for i in normals if rows[i]["fold"] != k],
                "test": [i for i in normals if rows[i]["fold"] == k] + defects,
            }
    elif name == "texturead":  # photo-disjoint cross-fitting within each subclass
        for sub in sorted({r["cls"] for r in rows}, key=int):
            for k in sorted({r["fold"] for r in rows}):
                tasks[f"class{sub}_fold{k}"] = {
                    "pool": [i for i, r in enumerate(rows)
                             if r["cls"] == sub and r["fold"] != k and r["label"] == 0],
                    "test": [i for i, r in enumerate(rows) if r["cls"] == sub and r["fold"] == k],
                }
    elif name == "mvtec":  # one task per category, released split
        for cat in sorted({r["cls"] for r in rows}):
            tasks[cat] = {
                "pool": [i for i, r in enumerate(rows) if r["cls"] == cat and r["split"] == "train"],
                "test": [i for i, r in enumerate(rows) if r["cls"] == cat and r["split"] == "test"],
            }
    if cfg.get("limit_test"):
        for t in tasks.values():
            lab = defaultdict(list)
            for i in t["test"]:
                lab[rows[i]["label"]].append(i)
            t["test"] = [i for v in lab.values() for i in v[: cfg["limit_test"]]]
    return tasks


# ----------------------------------------------------------------------------- #
# features and scoring for one (dataset, extractor, scale)
# ----------------------------------------------------------------------------- #
@torch.no_grad()
def tokens(paths, crop, extractor, boxes, size, bs):
    n_t = len(boxes)
    per = max(1, bs // n_t)
    for i in range(0, len(paths), per):
        chunk = paths[i : i + per]
        x = torch.cat([tiles_of(load(p, crop), boxes, size) for p in chunk])
        f = extractor(x)
        for j in range(len(chunk)):
            yield f[j * n_t : (j + 1) * n_t]


def run_scale(ds, rows, tasks, crop, sc, ext_name, extractor, cfg, seeds, raw_dir):
    out_dir = raw_dir / ds / f"{ext_name}__{sc['name']}"
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [(t, s) for t in tasks for s in seeds if not (out_dir / f"{t}_s{s}.npz").exists()]
    if not todo:
        return
    boxes = tile_boxes(sc["grid"], sc.get("overlap", 0.0))
    size, bs = sc["input_size"], sc.get("batch_size", cfg["batch_size"])
    mem, map_res = cfg["memory"], cfg["eval"]["map_res"]
    t0 = time.time()

    # bank + calibration split and memory bank per (task, seed)
    banks, need = {}, defaultdict(set)  # need: image index -> {(task, seed) to score it}
    calib_of, test_of = {}, {}
    for t, s in todo:
        gen = torch.Generator().manual_seed(s)
        pool = tasks[t]["pool"]
        perm = [pool[j] for j in torch.randperm(len(pool), generator=gen).tolist()]
        n_cal = max(5, int(round(cfg["calibration_fraction"] * len(perm))))
        calib, bank_imgs = perm[:n_cal], perm[n_cal:][: mem["n_images"]]
        picks = []
        for f in tokens([rows[i]["image"] for i in bank_imgs], crop, extractor, boxes, size, bs):
            f = f.reshape(-1, f.shape[-1])
            if mem["patches_per_tile"] <= 0:  # exhaustive bank: keep every token
                picks.append(f.half().cpu())
                continue
            n_keep = mem["patches_per_tile"] * len(boxes)
            picks.append(f[torch.randint(0, f.shape[0], (n_keep,), generator=gen)].half().cpu())
        bank = torch.cat(picks)
        if mem["coreset_size"] > 0 and bank.shape[0] > mem["coreset_size"]:
            bank = bank[torch.randperm(bank.shape[0], generator=gen)[: mem["coreset_size"]]]
        banks[(t, s)] = bank.to(DEVICE).float()
        calib_of[(t, s)], test_of[(t, s)] = calib, tasks[t]["test"]
        for i in calib + tasks[t]["test"]:
            need[i].add((t, s))

    # score every needed image once against every bank that needs it
    scores, maps = defaultdict(dict), defaultdict(dict)  # (t, s) -> {image index: value}
    order = sorted(need)
    for i, f in zip(order, tokens([rows[i]["image"] for i in order], crop, extractor, boxes,
                                  size, bs), strict=True):
        g, d = f.shape[1], f.shape[-1]
        q = f.reshape(-1, d)
        for key in need[i]:
            dist = min_dist_to_memory(q, banks[key]).reshape(len(boxes), g, g)
            scores[key][i] = dist.max().item()
            maps[key][i] = stitch(dist, boxes, map_res).half().cpu().numpy()

    for t, s in todo:
        key = (t, s)
        cm = np.stack([maps[key][i] for i in calib_of[key]]).astype(np.float32)
        np.savez_compressed(
            out_dir / f"{t}_s{s}.npz",
            test_idx=np.array(test_of[key]),
            test_score=np.array([scores[key][i] for i in test_of[key]]),
            test_map=np.stack([maps[key][i] for i in test_of[key]]),
            calib_score=np.array([scores[key][i] for i in calib_of[key]]),
            calib_map_mean=np.float32(cm.mean()), calib_map_std=np.float32(cm.std()),
        )
    print(f"  [{ds}] {ext_name}__{sc['name']}: {len(todo)} task-seeds, {len(order)} images "
          f"({time.time() - t0:.0f}s)", flush=True)


def parts_of(m: dict) -> list[tuple[str, str]]:
    """(extractor, scale) pairs a method fuses: ``parts`` if given, else extractor x scales."""
    if "parts" in m:
        return [tuple(p) for p in m["parts"]]
    return [(m["extractor"], sc) for sc in m["scales"]]


# ----------------------------------------------------------------------------- #
# metrics
# ----------------------------------------------------------------------------- #
def evaluate(cfg, datasets, methods, seeds, raw_dir, out_dir):
    res = cfg["eval"]["pixel_res"]
    rows_out = []
    for ds, (rows, tasks, crop) in datasets.items():
        mask_cache = {}
        for m in methods:
            for t in tasks:
                for s in seeds:
                    files = [raw_dir / ds / f"{e}__{sc}" / f"{t}_s{s}.npz" for e, sc in parts_of(m)]
                    if not all(f.exists() for f in files):
                        continue
                    zs, zm, idx = [], [], None
                    for f in files:
                        z = np.load(f)
                        idx = z["test_idx"]
                        amap = torch.from_numpy(z["test_map"].astype(np.float32))[:, None]
                        amap = F.interpolate(amap, size=(res, res), mode="bilinear",
                                             align_corners=False)[:, 0].numpy()
                        if len(files) == 1:
                            zs.append(z["test_score"])
                            zm.append(amap)
                        else:  # label-free standardization on training normals
                            c = z["calib_score"]
                            zs.append((z["test_score"] - c.mean()) / (c.std() + 1e-8))
                            zm.append((amap - z["calib_map_mean"]) / (z["calib_map_std"] + 1e-8))
                    score, amap = np.mean(zs, 0), np.mean(zm, 0)
                    y = np.array([rows[i]["label"] for i in idx])
                    for i in idx:
                        if i not in mask_cache:
                            mask_cache[i] = load_mask(rows[i]["mask"], crop, res)
                    gt = np.stack([mask_cache[i] for i in idx])
                    rec = {"dataset": ds, "method": m["name"], "task": t, "seed": s,
                           "image_auroc": float(roc_auc_score(y, score)),
                           "image_auprc": float(average_precision_score(y, score)),
                           "pixel_auroc": float(roc_auc_score(gt.reshape(-1), amap.reshape(-1))),
                           "pro": pro_auc(amap, gt)}
                    for fpr in (1, 5):  # recall at the (100 - fpr)th percentile of test normals
                        thr = np.percentile(score[y == 0], 100 - fpr)
                        rec[f"recall_at_{fpr}fpr"] = float((score[y == 1] >= thr).mean())
                    if ds == "carinthia":  # each defect class vs the held-out normals
                        cls = np.array([rows[i]["cls"] for i in idx])
                        for c in sorted(set(cls[y == 1])):
                            sel = (y == 0) | (cls == c)
                            rec[f"auroc_class{c}"] = float(roc_auc_score(y[sel], score[sel]))
                    rows_out.append(rec)
            print(f"[{ds}] {m['name']}: done", flush=True)

    (out_dir / "results.json").write_text(json.dumps(rows_out, indent=2))
    keys = sorted({k for r in rows_out for k in r})
    with (out_dir / "results.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows_out)

    # summary: mean over tasks within a seed, then mean +- std over seeds
    lines = []
    metric_keys = [k for k in keys if k.startswith(("image_", "pixel_", "pro", "auroc_class", "recall_"))]
    for ds in datasets:
        lines.append(f"== {ds} ==")
        for m in methods:
            rs = [r for r in rows_out if r["dataset"] == ds and r["method"] == m["name"]]
            if not rs:
                continue
            cells = []
            for k in metric_keys:
                per_seed = []
                for s in seeds:
                    v = [r[k] for r in rs if r["seed"] == s and k in r and r[k] == r[k]]
                    if v:
                        per_seed.append(np.mean(v))
                if per_seed:
                    # over seeds when there are several, else over tasks (folds)
                    spread = (np.std(per_seed) if len(per_seed) > 1 else
                              np.std([r[k] for r in rs if k in r and r[k] == r[k]]))
                    cells.append(f"{k}={np.mean(per_seed):.3f}+-{spread:.3f}")
            lines.append(f"{m['name']} (tasks={len({r['task'] for r in rs})}, seeds={len({r['seed'] for r in rs})}): "
                         + "  ".join(cells))
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    cfg = yaml.safe_load(CONFIG_PATH.open())
    seed_everything(cfg["seed"])
    man = Path(cfg["output"]["dir"])
    out_dir = Path(cfg["run"]["dir"])
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    seeds = cfg["run"]["seeds"]
    scales = {sc["name"]: sc for sc in cfg["scales"]}
    methods = [m for m in cfg["methods"] if m.get("enabled", True)]

    datasets = {}
    for ds in cfg["run"]["datasets"]:
        rows = read_manifest(man / f"{ds}.csv")
        tasks = make_tasks(ds, rows, cfg["run"])
        datasets[ds] = (rows, tasks, cfg[ds].get("crop", 0))
        print(f"{ds}: {len(rows)} images, tasks={ {t: (len(v['pool']), len(v['test'])) for t, v in tasks.items()} }",
              flush=True)

    # run every (extractor, scale) any enabled method needs
    needed = sorted({part for m in methods for part in parts_of(m)})
    global RGB
    for ext_name in sorted({e for e, _ in needed}):
        extractor = build_extractor(cfg["extractors"][ext_name], DEVICE)
        for ds, (rows, tasks, crop) in datasets.items():
            RGB = bool(cfg[ds].get("rgb", False))
            ds_seeds = cfg[ds].get("seeds", seeds)
            for e, sc in needed:
                if e == ext_name:
                    run_scale(ds, rows, tasks, crop, scales[sc], ext_name, extractor,
                              cfg["run"], ds_seeds, raw_dir)
        del extractor
        torch.cuda.empty_cache()
    evaluate(cfg["run"], datasets, methods, seeds, raw_dir, out_dir)


if __name__ == "__main__":
    main()
