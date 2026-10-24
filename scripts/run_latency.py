#!/usr/bin/env python
"""Inspection latency: feature extraction and exact NN search per image versus memory size.

Timing does not depend on the memory's content, so banks are random fp16 tensors of the real
shapes (384-dim tokens, 1,369 tokens per 518 px image).  For each memory size the script times
the exact search of one image and of a batch of 32 images (chunked fp16 GEMM with an fp32 min,
as in run_bigbank.py), after warm-up, median of repeats; and the DINOv2 forward pass per image.
Argument-free: configs/latency.yaml (or SEMIMIL_CONFIG).  Writes <output.dir>/latency.json.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_patchcore import build_extractor  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(os.environ.get("SEMIMIL_CONFIG", REPO_ROOT / "configs" / "latency.yaml"))
DEVICE = "cuda"


@torch.no_grad()
def search(q, bank, b2, b_chunk, q_chunk=4096):
    out = torch.empty(len(q), device=DEVICE)
    for i in range(0, len(q), q_chunk):
        qq = q[i : i + q_chunk]
        q2 = (qq.float() ** 2).sum(1, keepdim=True)
        best = torch.full((len(qq),), float("inf"), device=DEVICE)
        for j in range(0, len(bank), b_chunk):
            d2 = (qq @ bank[j : j + b_chunk].T).float()
            d2.mul_(-2.0).add_(q2).add_(b2[None, j : j + b_chunk])
            best = torch.minimum(best, torch.amin(d2, 1))
            del d2
        out[i : i + q_chunk] = best.clamp_min_(0).sqrt_()
    return out


def timed(fn, reps):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t = time.time(); fn(); torch.cuda.synchronize(); ts.append(time.time() - t)
    return sorted(ts)[len(ts) // 2]


def main() -> None:
    cfg = yaml.safe_load(CONFIG_PATH.open())
    out = Path(cfg["output"]["dir"]); out.mkdir(parents=True, exist_ok=True)
    per, dim = cfg["tokens_per_image"], cfg["dim"]
    res = {"gpu": torch.cuda.get_device_name(0), "search": []}
    ext = build_extractor(cfg["model"], DEVICE)
    for b in (1, 32):
        x = torch.randn(b, 3, cfg["input_size"], cfg["input_size"], device="cpu")
        res[f"extract_ms_per_image_b{b}"] = 1000 * timed(lambda: ext(x), cfg["reps"]) / b
    del ext
    torch.cuda.empty_cache()
    for m in sorted(cfg["memory_tokens"]):
        bank = torch.randn(int(m), dim, device=DEVICE, dtype=torch.float16)
        b2 = (bank.float() ** 2).sum(1) if m <= 5_000_000 else torch.cat(
            [(bank[i : i + 2_000_000].float() ** 2).sum(1) for i in range(0, len(bank), 2_000_000)])
        row = {"memory_tokens": int(m), "memory_images": round(m / per, 1),
               "memory_gb_fp16": round(m * dim * 2 / 1e9, 2)}
        for b in (1, 32):
            q = torch.randn(b * per, dim, device=DEVICE, dtype=torch.float16)
            row[f"search_ms_per_image_b{b}"] = 1000 * timed(lambda: search(q, bank, b2, cfg["b_chunk"]), cfg["reps"]) / b
        res["search"].append(row)
        print(row, flush=True)
        del bank, b2
        torch.cuda.empty_cache()
    print({k: v for k, v in res.items() if k != "search"}, flush=True)
    (out / "latency.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
