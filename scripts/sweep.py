#!/usr/bin/env python
"""Ablations + multi-seed sweep for Track A (one-class MIL) on MIIC.

Runs the memory-bank detector across extractors and seeds, plus 1-D ablations of
the DINOv2 variant (memory size, top-k pooling, input resolution). Reports image
AUROC/AUPRC as mean +/- std over seeds. Test features are cached per
(extractor, resolution) so only the memory bank is rebuilt per run.

Reproducible & argument-free: reads ``configs/sweep.yaml``. Pin a free GPU:
    CUDA_VISIBLE_DEVICES=1 uv run python scripts/sweep.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_patchcore import (
    batched,
    build_extractor,
    list_images,
    load_image,
    min_dist_to_memory,
)

from semimil.utils import seed_everything

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "sweep.yaml"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def extract(paths, extractor, size, bs):
    out = []
    for batch in batched(paths, bs):
        x = torch.stack([load_image(p, size) for p in batch])
        f = extractor(x)
        out.append(f.reshape(f.shape[0], -1, f.shape[-1]).half().cpu())
    return torch.cat(out)


def build_bank(mem_feats, patches, coreset, gen):
    picks = [
        mem_feats[b, torch.randint(0, mem_feats.shape[1], (patches,), generator=gen)]
        for b in range(len(mem_feats))
    ]
    bank = torch.cat(picks)
    if bank.shape[0] > coreset:
        bank = bank[torch.randperm(bank.shape[0], generator=gen)[:coreset]]
    return bank.to(DEVICE).float()


def score(test_feats, memory, top_k):
    s = []
    for i in range(test_feats.shape[0]):
        d = min_dist_to_memory(test_feats[i].to(DEVICE).float(), memory)
        s.append(torch.topk(d, top_k).values.mean().item())
    return np.array(s)


def main() -> None:
    cfg = yaml.safe_load(CONFIG_PATH.open())
    raw = Path(cfg["data"]["raw_dir"])
    dft = cfg["default"]
    bs = dft["batch_size"]
    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    train_normal = list_images(sorted(raw.glob(cfg["data"]["train_normal_glob"])))
    test_norm = list_images(sorted(Path(raw, cfg["data"]["test_normal_dir"]).iterdir()))
    test_abn = list_images(sorted(Path(raw, cfg["data"]["test_abnormal_dir"]).iterdir()))
    if cfg["limit_test"]:
        test_norm, test_abn = test_norm[: cfg["limit_test"]], test_abn[: cfg["limit_test"]]
    y = np.array([0] * len(test_norm) + [1] * len(test_abn))
    print(f"device={DEVICE}  test={len(test_norm)}N/{len(test_abn)}A", flush=True)

    # --- build the list of run specs ---
    def spec(group, ext, seed, **kw):
        p = {"group": group, "extractor": ext, "seed": seed,
             "input_size": cfg["extractors"][ext]["input_size"], **dft}
        p.pop("batch_size", None)
        p.update(kw)
        return p

    specs = []
    es = cfg["grid"]["extractor_seeds"]
    for ext, seeds in es.items():
        for s in seeds:
            specs.append(spec("extractor", ext, s))
    asd = cfg["grid"]["ablation_seeds"]
    for n in cfg["grid"]["n_images"]:
        for s in asd:
            specs.append(spec("n_images", "dinov2", s, n_images=n))
    for k in cfg["grid"]["top_k"]:
        for s in asd:
            specs.append(spec("top_k", "dinov2", s, top_k=k))
    for r in cfg["grid"]["resolution"]:
        for s in asd:
            specs.append(spec("resolution", "dinov2", s, input_size=r))
    if cfg["smoke"]:
        specs = [spec("extractor", "dinov2", 0), spec("top_k", "dinov2", 0, top_k=5)]
    print(f"total runs: {len(specs)}", flush=True)

    extractor_cache: dict = {}
    test_cache: dict = {}
    # resume: keep already-computed runs, skip them
    results = []
    res_path = out_dir / "results.json"
    if res_path.exists():
        results = json.loads(res_path.read_text())
    keyf = ("group", "extractor", "seed", "input_size", "n_images", "top_k")
    done = {tuple(r[k] for k in keyf) for r in results}

    for i, sp in enumerate(specs):
        key = tuple(sp[k] for k in keyf)
        if key in done:
            print(f"[{i + 1}/{len(specs)}] skip (done)  {key}", flush=True)
            continue
        t0 = time.time()
        seed_everything(sp["seed"])
        gen = torch.Generator().manual_seed(sp["seed"])
        ext_id = sp["extractor"]
        if ext_id not in extractor_cache:
            extractor_cache[ext_id] = build_extractor(cfg["extractors"][ext_id], DEVICE)
        extractor = extractor_cache[ext_id]
        size = sp["input_size"]

        # cached test features per (extractor, resolution)
        tkey = (ext_id, size)
        if tkey not in test_cache:
            fn = extract(test_norm, extractor, size, bs)
            fa = extract(test_abn, extractor, size, bs)
            test_cache[tkey] = torch.cat([fn, fa])
        test_feats = test_cache[tkey]

        # memory bank (depends on seed + n_images)
        idx = torch.randperm(len(train_normal), generator=gen)[: sp["n_images"]]
        mem_paths = [train_normal[j] for j in idx]
        mem_feats = extract(mem_paths, extractor, size, bs)
        memory = build_bank(mem_feats, sp["patches_per_image"], sp["coreset_size"], gen)

        s = score(test_feats, memory, sp["top_k"])
        rec = {
            **{k: sp[k] for k in ("group", "extractor", "seed", "input_size", "n_images", "top_k")},
            "image_auroc": float(roc_auc_score(y, s)),
            "image_auprc": float(average_precision_score(y, s)),
        }
        results.append(rec)
        dt = time.time() - t0
        print(
            f"[{i + 1}/{len(specs)}] {sp['group']:<10} {ext_id:<14} seed={sp['seed']} "
            f"n={sp['n_images']} k={sp['top_k']} res={size}  "
            f"AUROC={rec['image_auroc']:.4f} AUPRC={rec['image_auprc']:.4f}  ({dt:.0f}s)",
            flush=True,
        )
        (out_dir / "results.json").write_text(json.dumps(results, indent=2))

    summarize(results, out_dir)


def summarize(results, out_dir) -> None:
    import csv
    from collections import defaultdict

    def agg(rows):
        a = np.array([r["image_auroc"] for r in rows])
        p = np.array([r["image_auprc"] for r in rows])
        return a.mean(), a.std(), p.mean(), p.std(), len(rows)

    lines = []
    print("\n=== SUMMARY (mean +/- std over seeds) ===")
    # extractor comparison
    by_ext = defaultdict(list)
    for r in results:
        if r["group"] == "extractor":
            by_ext[r["extractor"]].append(r)
    for ext, rows in by_ext.items():
        am, as_, pm, ps, n = agg(rows)
        line = f"extractor  {ext:<14}  AUROC {am:.3f}+-{as_:.3f}  AUPRC {pm:.3f}+-{ps:.3f}  (n={n})"
        print(line)
        lines.append(line)
    # 1-D ablations
    for group, key in [("n_images", "n_images"), ("top_k", "top_k"), ("resolution", "input_size")]:
        by_val = defaultdict(list)
        for r in results:
            if r["group"] == group:
                by_val[r[key]].append(r)
        for val, rows in sorted(by_val.items()):
            am, as_, pm, ps, n = agg(rows)
            m = f"AUROC {am:.3f}+-{as_:.3f} AUPRC {pm:.3f}+-{ps:.3f}"
            line = f"{group:<9} {key}={val:<5} {m} n={n}"
            print(line)
            lines.append(line)
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")

    with (out_dir / "results.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"\nsaved -> {out_dir}")


if __name__ == "__main__":
    main()
