#!/usr/bin/env python
"""Exhaustive memory bank: does covering every normal token fix the small-defect recall?

Diagnosis behind it: on MIIC the 1 % false-positive threshold is set by a handful of
normal test images that contain rare but normal structures (line ends, vias, wide
metal planes).  The paper's bank keeps 256 random tokens from each of 1,000 random
training images and then 100,000 of those at random, so rare structures are thinly
represented, score high, raise the threshold, and the smallest defects fall under it.

Here the bank holds *every* patch token of the training normals, up to all 23,688
images (about 32 M tokens of 384 dims), and nearest-neighbor search is exact, done as
chunked matrix products on the GPU (bank kept on the GPU in fp16 after centering on
its mean, squared norms and the final distances in fp32).  Banks of increasing size are nested prefixes of one
random image order, so a single pass reports every size.

Outputs under ``output.dir``: raw/n<images>.npz (test + calibration image scores and
37x37 maps), results.json/.csv, summary.txt.  Argument-free: ``configs/bigbank.yaml``.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from localization import pro_auc  # noqa: E402
from run_patchcore import build_extractor, list_images, load_image  # noqa: E402

from semimil.utils import seed_everything  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(os.environ.get("SEMIMIL_CONFIG", REPO_ROOT / "configs" / "bigbank.yaml"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = True


@torch.no_grad()
def extract(paths, extractor, size, bs):
    out = []
    for i in range(0, len(paths), bs):
        x = torch.stack([load_image(p, size) for p in paths[i : i + bs]])
        f = extractor(x)
        out.append(f.reshape(-1, f.shape[-1]).half().cpu())
    return torch.cat(out)


@torch.no_grad()
def prefix_nn(queries, bank, prefixes, q_chunk, b_chunk):
    """Exact NN distance of every query to bank[:p] for each prefix p -> {p: [Q]}.

    The bank lives on the GPU in fp16, centered on its mean (which keeps the
    ||q||^2 + ||b||^2 - 2 q.b expansion well conditioned); squared norms are fp32.
    """
    mu = bank[: min(len(bank), 2_000_000)].float().mean(0)
    gb = torch.empty(bank.shape, dtype=torch.float16, device=DEVICE)
    for i in range(0, len(bank), 1_000_000):
        gb[i : i + 1_000_000] = (bank[i : i + 1_000_000].float() - mu).half().to(DEVICE)
    b2 = torch.cat([(gb[i : i + 1_000_000].float() ** 2).sum(1) for i in range(0, len(gb), 1_000_000)])
    mu = mu.to(DEVICE)
    out = {p: torch.empty(len(queries)) for p in prefixes}
    for qs in range(0, len(queries), q_chunk):
        qf = queries[qs : qs + q_chunk].to(DEVICE).float() - mu
        q2 = (qf * qf).sum(1, keepdim=True)
        q = qf.half()
        best = torch.full((len(q),), float("inf"), device=DEVICE)
        pos, pi = 0, 0
        while pos < len(gb):
            while pi < len(prefixes) and prefixes[pi] <= pos:  # snapshot finished prefixes
                out[prefixes[pi]][qs : qs + len(q)] = best.sqrt().cpu()
                pi += 1
            end = min(pos + b_chunk, len(gb))
            if pi < len(prefixes):
                end = min(end, prefixes[pi])  # never let a chunk cross a prefix boundary
            d2 = (q @ gb[pos:end].T).float()  # in place: one fp32 block, no extra temporaries
            d2.mul_(-2.0).add_(q2).add_(b2[pos:end][None, :])
            best = torch.minimum(best, d2.min(1).values.clamp_min(0))
            del d2
            pos = end
        while pi < len(prefixes):
            out[prefixes[pi]][qs : qs + len(q)] = best.sqrt().cpu()
            pi += 1
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
    mask_dir = Path(raw, cfg["data"]["test_mask_dir"])
    if cfg.get("limit_test"):  # smoke test only
        test_norm, test_abn = test_norm[: cfg["limit_test"]], test_abn[: cfg["limit_test"]]
    gen = torch.Generator().manual_seed(cfg["seed"])  # same calibration set as run_tiled/selfref
    perm = torch.randperm(len(train), generator=gen).tolist()
    n_cal = cfg["calibration"]["n_images"]
    calib, pool = [train[j] for j in perm[:n_cal]], [train[j] for j in perm[n_cal:]]
    order = torch.randperm(len(pool), generator=torch.Generator().manual_seed(cfg["bank_seed"])).tolist()
    pool = [pool[j] for j in order]
    sizes = [n if n > 0 else len(pool) for n in cfg["bank_images"]]
    size, bs = cfg["input_size"], cfg["batch_size"]
    print(f"device={DEVICE} pool={len(pool)} calib={len(calib)} test={len(test_norm)}N/{len(test_abn)}A "
          f"bank sizes (images)={sizes}", flush=True)

    extractor = build_extractor(cfg["model"], DEVICE)
    t0 = time.time()
    bank = extract(pool[: max(sizes)], extractor, size, bs)
    per_img = bank.shape[0] // max(sizes)
    print(f"bank tokens {tuple(bank.shape)} ({per_img}/image, {time.time() - t0:.0f}s)", flush=True)
    t0 = time.time()
    q_test = extract(test_norm + test_abn, extractor, size, bs)
    q_cal = extract(calib, extractor, size, bs)
    g = int(round(per_img ** 0.5))
    prefixes = [n * per_img for n in sizes]
    nn = prefix_nn(torch.cat([q_test, q_cal]), bank, prefixes, cfg["q_chunk"], cfg["b_chunk"])
    print(f"exact NN over {len(prefixes)} prefixes ({time.time() - t0:.0f}s)", flush=True)
    if cfg.get("check"):  # verify the chunked fp16 search against direct fp32 cdist
        qs = torch.cat([q_test, q_cal])[:2048].float().to(DEVICE)
        ref = torch.cdist(qs, bank[: prefixes[0]].float().to(DEVICE)).min(1).values.cpu()
        err = (nn[prefixes[0]][:2048] - ref).abs()
        print(f"check vs cdist: max abs err {err.max():.4f}, mean {err.mean():.5f}, "
              f"mean dist {ref.mean():.3f}", flush=True)

    n_t = len(test_norm) + len(test_abn)
    for n, p in zip(sizes, prefixes, strict=True):
        d = nn[p].reshape(-1, g, g)
        np.savez_compressed(out_dir / "raw" / f"n{n}.npz", test_map=d[:n_t].half().numpy(),
                            calib_map=d[n_t:].half().numpy())

    # metrics
    n_norm = len(test_norm)
    y = np.array([0] * n_norm + [1] * len(test_abn))
    res = cfg["eval"]["pixel_res"]
    mpaths = [mask_dir / f"{p.stem}_mask.jpg" for p in test_abn]
    areas = np.array([(np.asarray(Image.open(m).convert("L")) > 127).mean() * 100 for m in mpaths])
    masks = np.concatenate([np.zeros((n_norm, res, res), np.uint8), np.stack([
        (np.asarray(Image.open(m).convert("L").resize((res, res), Image.NEAREST)) > 127).astype(np.uint8)
        for m in mpaths])])
    rows = []
    for n in sizes:
        z = np.load(out_dir / "raw" / f"n{n}.npz")
        tm = z["test_map"].astype(np.float32)
        score = tm.reshape(len(tm), -1).max(1)
        amap = F.interpolate(torch.from_numpy(tm)[:, None], size=(res, res), mode="bilinear",
                             align_corners=False)[:, 0].numpy()
        normal, abn = score[:n_norm], score[n_norm:]
        thr = float(np.percentile(normal, 99))
        r = {"bank_images": n, "bank_tokens": n * per_img,
             "image_auroc": float(roc_auc_score(y, score)),
             "image_auprc": float(average_precision_score(y, score)),
             "recall_at_1fpr": float((abn >= thr).mean()), "threshold": thr,
             "pixel_auroc": float(roc_auc_score(masks.reshape(-1), amap.reshape(-1))),
             "pro": pro_auc(amap, masks)}
        for lo, hi, bn in cfg["eval"]["size_bins"]:
            sel = (areas >= lo) & (areas < hi)
            r[f"recall_{bn}"] = float((abn[sel] >= thr).mean())
            r[f"auroc_{bn}"] = float(roc_auc_score(np.r_[np.zeros(n_norm), np.ones(sel.sum())],
                                                   np.r_[normal, abn[sel]]))
        rows.append(r)
        print("  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()),
              flush=True)
    (out_dir / "results.json").write_text(json.dumps(rows, indent=2))
    with (out_dir / "results.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
