#!/usr/bin/env python
"""Greedy coreset versus random subsampling versus the exhaustive memory bank.

Question a reviewer will ask: PatchCore reduces its memory with a greedy k-center
coreset, which is designed to keep outlying (rare) features.  Does such a coreset,
at a fraction of the size, cover the rare normal structures that set the 1 % false-
alarm threshold on MIIC as well as keeping every token does?

For one token representation this script
1. extracts every patch token of the 23,688 pool images (same calibration split and
   pool as run_bigbank.py),
2. runs PatchCore's approximate greedy coreset (random projection to 128 dims,
   10 random starting points, iterative farthest-point selection) over ALL pool
   tokens; the selection order is nested, so one run gives every budget,
3. draws uniform random subsets of the same budgets from the same tokens,
4. scores test and calibration images by exact NN distance to each bank (full
   384-dim tokens), and saves raw/<method>_k<K>.npz in the format of run_bigbank.py
   (test_map, calib_map), so the analysis scripts compare it with raw/n23688.npz.

Argument-free: configs/coreset_<rep>.yaml (or SEMIMIL_CONFIG).
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
from PIL import Image
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_bigbank import extract, prefix_nn  # noqa: E402
from run_patchcore import build_extractor, list_images  # noqa: E402

from semimil.utils import seed_everything  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(os.environ.get("SEMIMIL_CONFIG", REPO_ROOT / "configs" / "coreset_L8.yaml"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = True


@torch.no_grad()
def project(bank: torch.Tensor, dim: int, seed: int) -> torch.Tensor:
    """Random Gaussian projection (Johnson-Lindenstrauss), as PatchCore, kept on the GPU in fp16."""
    g = torch.Generator().manual_seed(seed)
    w = (torch.randn(bank.shape[1], dim, generator=g) / dim ** 0.5).to(DEVICE)
    out = torch.empty((len(bank), dim), dtype=torch.float16, device=DEVICE)
    for i in range(0, len(bank), 1_000_000):
        out[i : i + 1_000_000] = (bank[i : i + 1_000_000].to(DEVICE).float() @ w).half()
    return out


@torch.no_grad()
def greedy_coreset(x: torch.Tensor, k: int, n_start: int, seed: int, log_every: int,
                   ckpt: Path | None = None) -> np.ndarray:
    """PatchCore's approximate greedy coreset on projected features x [N, d] (GPU, fp16)."""
    n = len(x)
    x2 = torch.empty(n, device=DEVICE)
    for i in range(0, n, 2_000_000):
        x2[i : i + 2_000_000] = (x[i : i + 2_000_000].float() ** 2).sum(1)

    def dist2(j: int) -> torch.Tensor:  # squared distance of every token to token j
        d = (x @ x[j]).float()
        return d.mul_(-2.0).add_(x2).add_(x2[j]).clamp_min_(0)

    g = torch.Generator().manual_seed(seed)
    start = torch.randint(0, n, (n_start,), generator=g).tolist()
    mind = torch.zeros(n, device=DEVICE)
    for j in start:  # PatchCore: mean distance to the starting points
        mind.add_(dist2(j).sqrt_())
    mind.div_(n_start)
    sel = torch.empty(k, dtype=torch.long, device=DEVICE)
    t0 = time.time()
    for i in range(k):
        j = int(torch.argmax(mind))
        sel[i] = j
        torch.minimum(mind, dist2(j).sqrt_(), out=mind)
        if (i + 1) % log_every == 0 or i + 1 == k:
            el = time.time() - t0
            print(f"  greedy {i + 1}/{k}  {el:.0f}s  ({1000 * el / (i + 1):.2f} ms/iter, "
                  f"eta {el / (i + 1) * (k - i - 1) / 60:.0f} min)", flush=True)
            if ckpt is not None:
                np.save(ckpt, sel[: i + 1].cpu().numpy())
    return sel.cpu().numpy()


def summarize(tm: np.ndarray, n_norm: int, areas: np.ndarray) -> dict:
    score = tm.reshape(len(tm), -1).max(1)
    y = np.r_[np.zeros(n_norm), np.ones(len(score) - n_norm)]
    normal, abn = score[:n_norm], score[n_norm:]
    thr = float(np.percentile(normal, 99))
    small = areas < 0.1
    return {"image_auroc": float(roc_auc_score(y, score)), "recall_at_1fpr": float((abn >= thr).mean()),
            "recall_small": float((abn[small] >= thr).mean()), "threshold": thr}


def main() -> None:
    cfg = yaml.safe_load(CONFIG_PATH.open())
    seed_everything(cfg["seed"])
    raw = Path(cfg["data"]["raw_dir"])
    out_dir = Path(cfg["output"]["dir"])
    (out_dir / "raw").mkdir(parents=True, exist_ok=True)
    train = list_images(sorted(raw.glob(cfg["data"]["train_normal_glob"])))
    test_norm = list_images(sorted(Path(raw, cfg["data"]["test_normal_dir"]).iterdir()))
    test_abn = list_images(sorted(Path(raw, cfg["data"]["test_abnormal_dir"]).iterdir()))
    mask_dir = Path(raw, cfg["data"]["test_mask_dir"])
    if cfg.get("limit_test"):  # smoke test only
        test_norm, test_abn = test_norm[: cfg["limit_test"]], test_abn[: cfg["limit_test"]]
    gen = torch.Generator().manual_seed(cfg["seed"])  # same calibration set as run_bigbank
    perm = torch.randperm(len(train), generator=gen).tolist()
    n_cal = cfg["calibration"]["n_images"]
    calib, pool = [train[j] for j in perm[:n_cal]], [train[j] for j in perm[n_cal:]]
    if cfg.get("pool_images"):  # smoke test only
        pool = pool[: cfg["pool_images"]]
    size, bs = cfg["input_size"], cfg["batch_size"]
    print(f"device={DEVICE} pool={len(pool)} calib={len(calib)} test={len(test_norm)}N/{len(test_abn)}A",
          flush=True)

    extractor = build_extractor(cfg["model"], DEVICE)
    t0 = time.time()
    bank = extract(pool, extractor, size, bs)
    q = torch.cat([extract(test_norm + test_abn, extractor, size, bs), extract(calib, extractor, size, bs)])
    per_img = bank.shape[0] // len(pool)
    g = int(round(per_img ** 0.5))
    del extractor
    torch.cuda.empty_cache()
    print(f"tokens: bank {tuple(bank.shape)}, queries {tuple(q.shape)} ({time.time() - t0:.0f}s)", flush=True)

    cs = cfg["coreset"]
    ks = sorted(int(round(f * len(bank))) if f < 1 else int(f) for f in cs["budgets"])
    t0 = time.time()
    xp = project(bank, cs["proj_dim"], cs["seed"])
    order = greedy_coreset(xp, max(ks), cs["n_start"], cs["seed"], cs["log_every"],
                           out_dir / "greedy_order_partial.npy")
    del xp
    torch.cuda.empty_cache()
    np.save(out_dir / "greedy_order.npy", order)
    print(f"greedy coreset of {max(ks)} tokens ({time.time() - t0:.0f}s)", flush=True)
    rnd = torch.randperm(len(bank), generator=torch.Generator().manual_seed(cs["seed"]))[: max(ks)].numpy()

    n_t, n_norm = len(test_norm) + len(test_abn), len(test_norm)
    areas = np.array([(np.asarray(Image.open(mask_dir / f"{p.stem}_mask.jpg").convert("L")) > 127).mean() * 100
                      for p in test_abn])
    rows = []
    for method, idx in (("greedy", order), ("random", rnd)):
        t0 = time.time()
        sub = bank[torch.from_numpy(idx)]
        nn = prefix_nn(q, sub, ks, cfg["q_chunk"], cfg["b_chunk"])
        for k in ks:
            d = nn[k].reshape(-1, g, g)
            np.savez_compressed(out_dir / "raw" / f"{method}_k{k}.npz", test_map=d[:n_t].half().numpy(),
                                calib_map=d[n_t:].half().numpy())
            r = {"method": method, "k": k, "fraction": k / len(bank),
                 **summarize(d[:n_t].numpy().astype(np.float32), n_norm, areas)}
            rows.append(r)
            print("  ".join(f"{a}={b:.4f}" if isinstance(b, float) else f"{a}={b}" for a, b in r.items()),
                  flush=True)
        print(f"{method}: NN for {len(ks)} budgets ({time.time() - t0:.0f}s)", flush=True)
    (out_dir / "results.json").write_text(json.dumps(rows, indent=2))

    if cfg.get("bench_full_n"):  # smoke test: time one greedy iteration at the full bank size
        n = int(cfg["bench_full_n"])
        x = torch.randn(n, cs["proj_dim"], device=DEVICE).half()
        greedy_coreset(x, 300, 2, 0, 300)


if __name__ == "__main__":
    main()
