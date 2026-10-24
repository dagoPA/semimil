#!/usr/bin/env python
"""Inspection memory under process drift: Texture-AD wafer photographs in acquisition order.

Texture-AD's 14 wafer subclasses were photographed in sequence over three days, and every
file name carries the capture time.  Per subclass, photographs are ordered by time:

  memory      <- normal crops of the first n_memory photographs
  calibration <- normal crops of the next n_calib photographs (never in the memory); they
                 set the fused z-scale and the alarm threshold (their maximum)
  stream      <- every later photograph, in time order.  For photograph t:
      score all its crops with the current memory and record, at the fixed threshold,
      the false-alarm rate on its normal crops and the sensitivity on its defective crops
      (AUROC per photograph as well); then update the memory with one of the policies:
        frozen  nothing is added
        alarm   the normal crops that raised an alarm are added (an inspector reviews
                every alarm; defective crops that alarmed are rejected, never added)
        random  as many normal crops as 'alarm' added, chosen at random
        all     every normal crop of the photograph is added (inspects everything)
Reported per subclass and policy: mean false-alarm rate and sensitivity over the stream, the
number of crops an inspector had to review, and the per-photograph trajectory (control chart).

Detector: the proposed one (block-8 tokens and 3x3-pooled block-10 tokens of DINOv2 ViT-S/14,
exact nearest neighbor, image score = max patch distance, fused by calibration z-scores).
Argument-free: configs/stream_texturead.yaml (or SEMIMIL_CONFIG).
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
import yaml
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_patchcore import build_extractor, load_image  # noqa: E402

from semimil.utils import seed_everything  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(os.environ.get("SEMIMIL_CONFIG", REPO_ROOT / "configs" / "stream_texturead.yaml"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
POLICIES = ("frozen", "alarm", "random", "all")


@torch.no_grad()
def tokens(paths, extractor, size, bs):
    out = []
    for i in range(0, len(paths), bs):
        f = extractor(torch.stack([load_image(p, size) for p in paths[i : i + bs]]))
        out.append(f.reshape(f.shape[0], -1, f.shape[-1]).half())
    return torch.cat(out)  # [n, tokens, D] on the GPU


@torch.no_grad()
def image_scores(q, mem, q_chunk=4096, m_chunk=100_000):
    """Max over tokens of the exact NN distance to the memory: q [n, T, D], mem [M, D] -> [n].

    Both sides are chunked (the memory reaches a few million tokens under the 'all' policy);
    fp16 products with an fp32 expansion, centered on the memory mean, as in run_bigbank.py.
    """
    n, t, d = q.shape
    mu = mem[: min(len(mem), 1_000_000)].float().mean(0)
    m = (mem.float() - mu).half()
    m2 = (m.float() ** 2).sum(1)
    flat = q.reshape(-1, d)
    out = torch.empty(len(flat), device=DEVICE)
    for i in range(0, len(flat), q_chunk):
        x = flat[i : i + q_chunk].float() - mu
        x2 = (x * x).sum(1, keepdim=True)
        x = x.half()
        best = torch.full((len(x),), float("inf"), device=DEVICE)
        for j in range(0, len(m), m_chunk):
            d2 = (x @ m[j : j + m_chunk].T).float()
            d2.mul_(-2.0).add_(x2).add_(m2[None, j : j + m_chunk])
            best = torch.minimum(best, torch.amin(d2, 1))
            del d2
        out[i : i + q_chunk] = best.clamp_min_(0).sqrt_()
    return out.view(n, t).max(1).values


def auroc(y, s):
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def main() -> None:
    cfg = yaml.safe_load(CONFIG_PATH.open())
    seed_everything(cfg["seed"])
    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(cfg["data"]["manifest"])))
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r["cls"]][r["photo"]].append(r)
    nm, nc = cfg["n_memory"], cfg["n_calib"]
    classes = [c for c in sorted(by, key=int) if len(by[c]) >= nm + nc + cfg["min_stream"]]
    if cfg.get("classes"):
        classes = [c for c in classes if c in {str(x) for x in cfg["classes"]}]
    print(f"device={DEVICE} subclasses={classes}", flush=True)
    extractors = {r["name"]: build_extractor(r["model"], DEVICE) for r in cfg["representations"]}
    size, bs = cfg["input_size"], cfg["batch_size"]
    rng = np.random.RandomState(cfg["seed"])
    summary, traj = {}, {}
    t0 = time.time()
    for c in classes:
        photos = sorted(by[c], key=lambda p: p.split("_")[1])  # capture time in the file name
        mem_ph, cal_ph, stream = photos[:nm], photos[nm : nm + nc], photos[nm + nc :]
        crops = [r for p in photos for r in by[c][p]]
        idx = {r["image"]: i for i, r in enumerate(crops)}
        feats = {n: tokens([r["image"] for r in crops], e, size, bs) for n, e in extractors.items()}
        lab = np.array([int(r["label"]) for r in crops])
        sel = lambda phs, label: [idx[r["image"]] for p in phs for r in by[c][p] if int(r["label"]) == label]
        mem0, cal = sel(mem_ph, 0), sel(cal_ph, 0)
        res = {}
        for pol in POLICIES:
            mem = list(mem0)
            # calibration: fused z-scale and threshold from the calibration normals (fixed for the stream)
            def fused(ids):
                return torch.stack([(image_scores(feats[n][ids], feats[n][mem].reshape(-1, feats[n].shape[-1]))
                                     - mu[n]) / sd[n] for n in feats]).mean(0)
            mu, sd = {}, {}
            for n in feats:
                s = image_scores(feats[n][cal], feats[n][mem0].reshape(-1, feats[n].shape[-1]))
                mu[n], sd[n] = s.mean(), s.std()
            per_photo = []
            for p in stream:
                # recalibrate on the held-out calibration normals with the current memory: distances
                # shrink as the memory grows, so a fixed threshold would confound the policies
                thr = float(fused(cal).max())
                ids = [idx[r["image"]] for r in by[c][p]]
                s = fused(ids).cpu().numpy()
                y = lab[ids]
                alarm = s > thr
                n_idx = [i for i, a, yy in zip(ids, alarm, y) if yy == 0]
                fa_idx = [i for i, a, yy in zip(ids, alarm, y) if yy == 0 and a]
                per_photo.append({"photo": p, "n_normal": int((y == 0).sum()), "n_defect": int(y.sum()),
                                  "fpr": float(alarm[y == 0].mean()) if (y == 0).any() else float("nan"),
                                  "sens": float(alarm[y == 1].mean()) if y.any() else float("nan"),
                                  "auroc": auroc(y, s), "mean_normal_z": float(s[y == 0].mean()) if (y == 0).any() else float("nan"),
                                  "threshold": thr,
                                  "reviewed": int(alarm.sum()) if pol != "all" else len(ids),
                                  "memory_crops": len(mem)})
                if pol == "alarm":
                    mem += fa_idx
                elif pol == "random":
                    mem += list(rng.choice(n_idx, min(len(fa_idx), len(n_idx)), replace=False)) if fa_idx else []
                elif pol == "all":
                    mem += n_idx
            pp = per_photo
            w = lambda k, wk: float(np.nansum([x[k] * x[wk] for x in pp]) / max(1, sum(x[wk] for x in pp if x[k] == x[k])))
            added = len(mem) - len(mem0)
            res[pol] = {"fpr": w("fpr", "n_normal"), "sens": w("sens", "n_defect"),
                        "auroc": float(np.nanmean([x["auroc"] for x in pp])),
                        # random: the inspector also confirms the crops it adds
                        "reviewed": int(sum(x["reviewed"] for x in pp)) + (added if pol == "random" else 0),
                        "added": added}
            traj[f"{c}/{pol}"] = per_photo
        summary[c] = {"photos": len(photos), "stream_photos": len(stream), "memory_crops0": len(mem0),
                      "calib_crops": len(cal), **res}
        print(f"subclass {c:>2}: {len(stream)} stream photos, memory {len(mem0)} crops  " + "  ".join(
            f"{p}: FPR {res[p]['fpr']:.3f} sens {res[p]['sens']:.3f} AUROC {res[p]['auroc']:.3f} "
            f"reviewed {res[p]['reviewed']} added {res[p]['added']}" for p in POLICIES)
              + f"  ({time.time() - t0:.0f}s)", flush=True)
        del feats
        torch.cuda.empty_cache()
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
        (out_dir / "trajectories.json").write_text(json.dumps(traj))
    for p in POLICIES:
        v = np.array([[summary[c][p][k] for k in ("fpr", "sens", "auroc")] for c in summary])
        r = sum(summary[c][p]["reviewed"] for c in summary)
        print(f"MEAN over {len(summary)} subclasses  {p:6s}: FPR {v[:, 0].mean():.3f}  sens {v[:, 1].mean():.3f}  "
              f"AUROC {np.nanmean(v[:, 2]):.3f}  reviewed {r}", flush=True)


if __name__ == "__main__":
    main()
