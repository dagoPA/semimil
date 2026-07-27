#!/usr/bin/env python
"""Deployment budget of the one-class detector on MIIC: how little does it need?

Two label-free questions that decide whether the method is adoptable on a line:

(a) *Normal-image budget.* Image AUROC as a function of the number of normal
    images k in the memory bank, down to k=1. The existing ablation only reaches
    k=200 and is already flat there, so the informative regime is below it.

(b) *Contamination tolerance.* In practice nobody guarantees a clean "normal"
    pool: images are collected off the line and a fraction eps of them are
    defective. We contaminate the pool at eps in [0, 10%] with either synthetic
    anomalies (keeping the full 116-positive test set) or real ones (held out
    from the test set, so contaminants never leak into evaluation), and compare
    the raw bank against a label-free self-cleaning pass: score every pool image
    against the bank while masking out its own patches, drop the most anomalous
    fraction, rebuild. The operator does not know eps, so the drop fraction is
    fixed -- which also prices the cost of self-cleaning a pool that was clean.

Features are extracted once and cached, so every sweep point is only a
nearest-neighbour search.

Reproducible & argument-free: reads ``configs/deployment_budget.yaml``.

    CUDA_VISIBLE_DEVICES=1 uv run python scripts/deployment_budget.py
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
CONFIG_PATH = REPO_ROOT / "configs" / "deployment_budget.yaml"


# ----------------------------------------------------------------------------- #
# features (extract once, cache to disk)
# ----------------------------------------------------------------------------- #
@torch.no_grad()
def extract(paths, extractor, size, bs):
    """[N images] -> [N, T, D] half-precision patch tokens on CPU."""
    out = []
    for batch in batched(paths, bs):
        x = torch.stack([load_image(p, size) for p in batch])
        f = extractor(x)
        out.append(f.reshape(f.shape[0], -1, f.shape[-1]).half().cpu())
    return torch.cat(out)


def extract_cached(tag, paths, extractor, size, bs, cache_dir):
    path = cache_dir / f"{tag}.pt"
    if path.exists():
        feats = torch.load(path, map_location="cpu")
        if feats.shape[0] == len(paths):
            print(f"[feat] {tag}: cached {tuple(feats.shape)}", flush=True)
            return feats
        print(f"[feat] {tag}: cache stale ({feats.shape[0]} != {len(paths)}), re-extracting")
    t0 = time.time()
    feats = extract(paths, extractor, size, bs)
    torch.save(feats, path)
    print(f"[feat] {tag}: {tuple(feats.shape)}  ({time.time() - t0:.1f}s)", flush=True)
    return feats


# ----------------------------------------------------------------------------- #
# memory bank
# ----------------------------------------------------------------------------- #
def build_bank(feats, patches, coreset, gen, device):
    """[N,T,D] -> (bank [K,D] on device, owner [K] pool-image index).

    ``owner`` keeps patch provenance so a pool image can be scored against the
    bank without matching itself (needed by the self-cleaning pass).
    """
    n, t, _ = feats.shape
    idx = torch.randint(0, t, (n, patches), generator=gen)
    bank = torch.stack([feats[i, idx[i]] for i in range(n)]).reshape(n * patches, -1)
    owner = torch.arange(n).repeat_interleave(patches)
    if bank.shape[0] > coreset:
        sel = torch.randperm(bank.shape[0], generator=gen)[:coreset]
        bank, owner = bank[sel], owner[sel]
    return bank.to(device).float(), owner.to(device)


def score_test(test_feats, memory, top_k, device):
    """Image score = mean of the top-k patch NN distances (the paper's rule)."""
    s = np.empty(test_feats.shape[0])
    for i in range(test_feats.shape[0]):
        d = min_dist_to_memory(test_feats[i].to(device).float(), memory)
        s[i] = torch.topk(d, top_k).values.mean().item()
    return s


@torch.no_grad()
def score_pool_excluding_self(pool_feats, memory, owner, probe, top_k, gen, device):
    """Score each pool image against the bank, masking out its own patches.

    Without the mask a pool image matches the copies of its own patches that sit
    in the bank at distance ~0, which makes every image look normal.
    """
    n, t, _ = pool_feats.shape
    idx = torch.randint(0, t, (n, probe), generator=gen)
    s = np.empty(n)
    for i in range(n):
        q = pool_feats[i, idx[i]].to(device).float()  # [probe, D]
        d = torch.cdist(q, memory)  # [probe, K]
        d[:, owner == i] = float("inf")
        nn = d.min(dim=1).values
        s[i] = torch.topk(nn, top_k).values.mean().item()
    return s


# ----------------------------------------------------------------------------- #
def main() -> None:
    cfg = yaml.safe_load(CONFIG_PATH.open())
    device = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if cfg["device"] == "auto"
        else cfg["device"]
    )
    seed_everything(cfg["seed"])
    raw = Path(cfg["data"]["raw_dir"])
    det, size, bs = cfg["detector"], cfg["model"]["input_size"], cfg["model"]["batch_size"]
    out_dir = Path(cfg["output"]["dir"])
    cache_dir = Path(cfg["output"]["cache_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device}  out={out_dir}", flush=True)

    # --- image lists ----------------------------------------------------------
    rng = np.random.default_rng(cfg["seed"])
    all_normal = list_images(sorted(raw.glob(cfg["data"]["train_normal_glob"])))
    all_syn = list_images(sorted(raw.glob(cfg["data"]["synthetic_glob"])))
    n_pool = cfg["pools"]
    normal_pool = [all_normal[i] for i in rng.permutation(len(all_normal))[: n_pool["n_normal"]]]
    syn_pool = [all_syn[i] for i in rng.permutation(len(all_syn))[: n_pool["n_synthetic"]]]
    test_norm = list_images(sorted(Path(raw, cfg["data"]["test_normal_dir"]).iterdir()))
    test_abn = list_images(sorted(Path(raw, cfg["data"]["test_abnormal_dir"]).iterdir()))
    print(f"pools: {len(normal_pool)} normal, {len(syn_pool)} synthetic | "
          f"test: {len(test_norm)}N/{len(test_abn)}A", flush=True)

    # real anomalies: a fixed half becomes contaminants, the other half stays in test
    ccfg = cfg["contamination"]
    perm = np.random.default_rng(ccfg["real_split_seed"]).permutation(len(test_abn))
    n_contam_real = round(ccfg["real_frac_contam"] * len(test_abn))
    real_contam_idx = np.sort(perm[:n_contam_real])
    real_test_idx = np.sort(perm[n_contam_real:])
    print(f"real anomalies: {len(real_contam_idx)} contaminants / "
          f"{len(real_test_idx)} kept in test", flush=True)

    # --- features (once) ------------------------------------------------------
    extractor = build_extractor(cfg["model"], device)
    f_normal = extract_cached("normal_pool", normal_pool, extractor, size, bs, cache_dir)
    f_syn = extract_cached("syn_pool", syn_pool, extractor, size, bs, cache_dir)
    f_tn = extract_cached("test_normal", test_norm, extractor, size, bs, cache_dir)
    f_ta = extract_cached("test_abnormal", test_abn, extractor, size, bs, cache_dir)

    # evaluation arms: full test set, and the reduced one for the real-contaminant arm
    test_full = torch.cat([f_tn, f_ta])
    y_full = np.array([0] * len(f_tn) + [1] * len(f_ta))
    test_half = torch.cat([f_tn, f_ta[real_test_idx]])
    y_half = np.array([0] * len(f_tn) + [1] * len(real_test_idx))

    # --- resume ---------------------------------------------------------------
    res_path = out_dir / "results.json"
    results = json.loads(res_path.read_text()) if res_path.exists() else []
    keyf = ("part", "source", "k", "rate", "clean", "seed")
    done = {tuple(r[k] for k in keyf) for r in results}

    def record(rec, y, s, t0):
        rec["image_auroc"] = float(roc_auc_score(y, s))
        rec["image_auprc"] = float(average_precision_score(y, s))
        rec["seconds"] = round(time.time() - t0, 1)
        results.append(rec)
        res_path.write_text(json.dumps(results, indent=2))
        tag = " ".join(f"{k}={rec[k]}" for k in keyf if rec[k] is not None)
        print(f"  {tag}  AUROC={rec['image_auroc']:.4f} "
              f"AUPRC={rec['image_auprc']:.4f}  ({rec['seconds']}s)", flush=True)

    # ==== (a) normal-image budget ============================================
    print("\n=== (a) normal-image budget ===", flush=True)
    for k in cfg["budget"]["k_images"]:
        for seed in cfg["budget"]["seeds"]:
            key = ("budget", None, k, None, False, seed)
            if key in done:
                continue
            t0 = time.time()
            gen = torch.Generator().manual_seed(10_000 * seed + k)
            idx = torch.randperm(len(f_normal), generator=gen)[:k]
            memory, _ = build_bank(
                f_normal[idx], det["patches_per_image"], det["coreset_size"], gen, device
            )
            s = score_test(test_full, memory, det["top_k"], device)
            record({"part": "budget", "source": None, "k": k, "rate": None,
                    "clean": False, "seed": seed, "bank": int(memory.shape[0])},
                   y_full, s, t0)
            del memory
            torch.cuda.empty_cache() if device == "cuda" else None

    # ==== (b) contamination tolerance ========================================
    print("\n=== (b) contamination tolerance ===", flush=True)
    pool_size = ccfg["pool_size"]
    sc = ccfg["self_clean"]
    for source in ccfg["sources"]:
        f_contam = f_syn if source == "synthetic" else f_ta[real_contam_idx]
        test_feats, y = (test_full, y_full) if source == "synthetic" else (test_half, y_half)
        for rate in ccfg["rates"]:
            n_bad = round(rate * pool_size)
            if n_bad > f_contam.shape[0]:
                print(f"  skip {source} rate={rate}: needs {n_bad} contaminants, "
                      f"only {f_contam.shape[0]} available", flush=True)
                continue
            for seed in ccfg["seeds"]:
                variants = [False, True] if sc["enabled"] else [False]
                if all(("contam", source, None, rate, c, seed) in done for c in variants):
                    continue
                gen = torch.Generator().manual_seed(20_000 * seed + int(rate * 1000))
                n_good = pool_size - n_bad
                gi = torch.randperm(len(f_normal), generator=gen)[:n_good]
                bi = torch.randperm(f_contam.shape[0], generator=gen)[:n_bad]
                pool = torch.cat([f_normal[gi], f_contam[bi]])
                # is_bad marks the contaminants, used only to report what self-cleaning caught
                is_bad = np.array([0] * n_good + [1] * n_bad)
                shuf = torch.randperm(pool.shape[0], generator=gen)
                pool, is_bad = pool[shuf], is_bad[shuf.numpy()]

                memory, owner = build_bank(
                    pool, det["patches_per_image"], det["coreset_size"], gen, device
                )
                base = {"part": "contam", "source": source, "k": None, "rate": rate,
                        "seed": seed, "n_contaminants": n_bad, "pool_size": pool_size}

                if ("contam", source, None, rate, False, seed) not in done:
                    t0 = time.time()
                    s = score_test(test_feats, memory, det["top_k"], device)
                    record({**base, "clean": False, "bank": int(memory.shape[0])}, y, s, t0)

                if sc["enabled"] and ("contam", source, None, rate, True, seed) not in done:
                    t0 = time.time()
                    ps = score_pool_excluding_self(
                        pool, memory, owner, sc["probe_patches"], det["top_k"], gen, device
                    )
                    n_drop = round(sc["drop_frac"] * pool.shape[0])
                    drop = np.argsort(-ps)[:n_drop]
                    keep = np.setdiff1d(np.arange(pool.shape[0]), drop)
                    caught = int(is_bad[drop].sum())
                    del memory, owner
                    mem2, _ = build_bank(
                        pool[keep], det["patches_per_image"], det["coreset_size"], gen, device
                    )
                    s = score_test(test_feats, mem2, det["top_k"], device)
                    record({**base, "clean": True, "bank": int(mem2.shape[0]),
                            "n_dropped": n_drop, "contaminants_caught": caught,
                            "recall_contaminants": (caught / n_bad) if n_bad else None},
                           y, s, t0)
                    del mem2
                else:
                    del memory, owner
                torch.cuda.empty_cache() if device == "cuda" else None

    print(f"\n{len(results)} runs -> {res_path}")
    summarize(results)


def summarize(results: list[dict]) -> None:
    """Print the two curves as mean +/- std over seeds."""
    def agg(rows):
        a = np.array([r["image_auroc"] for r in rows])
        return f"{a.mean():.4f} +/- {a.std(ddof=1) if len(a) > 1 else 0.0:.4f} (n={len(a)})"

    print("\n=== (a) normal-image budget (image AUROC) ===")
    bud = [r for r in results if r["part"] == "budget"]
    for k in sorted({r["k"] for r in bud}):
        print(f"  k={k:>5}  {agg([r for r in bud if r['k'] == k])}")

    print("\n=== (b) contamination tolerance (image AUROC) ===")
    con = [r for r in results if r["part"] == "contam"]
    for source in sorted({r["source"] for r in con}):
        print(f"  -- contaminants: {source}")
        for rate in sorted({r["rate"] for r in con if r["source"] == source}):
            for clean in (False, True):
                rows = [r for r in con if r["source"] == source
                        and r["rate"] == rate and r["clean"] == clean]
                if not rows:
                    continue
                lbl = "self-cleaned" if clean else "raw        "
                rec = [r["recall_contaminants"] for r in rows if r.get("recall_contaminants")]
                extra = f"  contaminant recall={np.mean(rec):.2f}" if rec else ""
                print(f"     eps={rate:<6} {lbl}  {agg(rows)}{extra}")


if __name__ == "__main__":
    main()
