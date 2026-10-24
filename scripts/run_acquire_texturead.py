#!/usr/bin/env python
"""Alarm-driven vs random acquisition of normals on Texture-AD wafers (photo-disjoint, many defects).

MIIC has 116 defective test images, too few to resolve a few points of sensitivity.  Texture-AD
has about 2,300 defective crops in the subclasses used here, so the closed loop of
scripts/run_acquire.py is repeated per task = (wafer subclass, photograph fold):
  pool         normal crops of the other fold's photographs, minus the calibration photographs
  calibration  normal crops of ~15 % of the other fold's photographs (never in the memory)
  test         every crop (normal and defective) of the fold's photographs
For each seed an initial memory of n0 = frac[0] x |pool| random pool crops is drawn; from it
  alarm   adds, each round, the pool crops with the highest fused z-score (confirmed normal)
  random  adds random pool crops
until frac[-1] x |pool|.  Detector: block-8 and pooled block-10 DINOv2 tokens, exact NN, image
score = max patch distance, z-scored on the calibration crops under the current memory, fused.
Per checkpoint: test AUROC, Sens@Spec 0.95 / 0.99 (threshold from the test normals), and the
fused test scores (for pooled analyses).  Writes <output.dir>/results.json and scores.npz.
Argument-free: configs/acquire_texturead.yaml (or SEMIMIL_CONFIG).
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
CONFIG_PATH = Path(os.environ.get("SEMIMIL_CONFIG", REPO_ROOT / "configs" / "acquire_texturead.yaml"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def tokens(paths, extractor, size, bs):
    out = []
    for i in range(0, len(paths), bs):
        f = extractor(torch.stack([load_image(p, size) for p in paths[i : i + bs]]))
        out.append(f.reshape(f.shape[0], -1, f.shape[-1]).half())
    return torch.cat(out)  # [n, T, D] on the GPU


@torch.no_grad()
def lower(dmin, q, q2, new, mu, q_chunk=8192, m_chunk=100_000):
    """dmin [Nq] <- min(dmin, NN distance of the centered queries q [Nq, D] to new [M, D])."""
    m = (new.float() - mu).half()
    m2 = (m.float() ** 2).sum(1)
    for i in range(0, len(q), q_chunk):
        x, x2 = q[i : i + q_chunk], q2[i : i + q_chunk, None]
        best = dmin[i : i + q_chunk]
        for j in range(0, len(m), m_chunk):
            d2 = (x @ m[j : j + m_chunk].T).float()
            d2.mul_(-2.0).add_(x2).add_(m2[None, j : j + m_chunk])
            best = torch.minimum(best, torch.amin(d2, 1).clamp_min_(0).sqrt_())
            del d2
        dmin[i : i + q_chunk] = best


def sens_at_spec(y, s, spec):
    thr = np.percentile(s[y == 0], 100 * spec)
    return float((s[y == 1] > thr).mean())


def main() -> None:
    cfg = yaml.safe_load(CONFIG_PATH.open())
    seed_everything(0)
    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(cfg["data"]["manifest"])))
    by = defaultdict(list)
    for r in rows:
        by[r["cls"]].append(r)
    classes = [c for c in sorted(by, key=int) if c not in {str(x) for x in cfg.get("skip_classes", [])}]
    if cfg.get("classes"):
        classes = [c for c in classes if c in {str(x) for x in cfg["classes"]}]
    reps = {r["name"]: build_extractor(r["model"], DEVICE) for r in cfg["representations"]}
    size, bs, fracs = cfg["input_size"], cfg["batch_size"], cfg["fractions"]
    results, scores = [], {}
    t0 = time.time()
    print(f"device={DEVICE} subclasses={classes} fractions={fracs} seeds={cfg['seeds']}", flush=True)
    for c in classes:
        crops = by[c]
        feats = {n: tokens([r["image"] for r in crops], e, size, bs) for n, e in reps.items()}
        per = next(iter(feats.values())).shape[1]
        lab = np.array([int(r["label"]) for r in crops])
        for fold in ("0", "1"):
            test = [i for i, r in enumerate(crops) if r["fold"] == fold]
            other = [i for i, r in enumerate(crops) if r["fold"] != fold and r["label"] == "0"]
            photos = sorted({crops[i]["photo"] for i in other})
            rng = np.random.RandomState(int(c) * 10 + int(fold))
            n_cal = max(1, int(round(cfg["calib_photo_frac"] * len(photos))))
            cal_ph = set(rng.choice(photos, n_cal, replace=False))
            cal = [i for i in other if crops[i]["photo"] in cal_ph]
            pool = [i for i in other if crops[i]["photo"] not in cal_ph]
            sizes = sorted({max(2, int(round(f * len(pool)))) for f in fracs})
            y = lab[test]
            if y.sum() < 5 or (y == 0).sum() < 20 or len(pool) < 40:
                print(f"skip {c}/{fold}: pool {len(pool)}, test normals {(y == 0).sum()}, defects {y.sum()}")
                continue
            for seed in cfg["seeds"]:
                init = list(np.random.RandomState(1000 + seed).choice(len(pool), sizes[0], replace=False))
                for strat in ("alarm", "random"):
                    order = [k for k in np.random.RandomState(2000 + seed).permutation(len(pool)) if k not in set(init)]
                    st = {}
                    for n, F in feats.items():
                        mu = F[[pool[k] for k in init]].reshape(-1, F.shape[-1]).float().mean(0)
                        q = {}
                        for part, ids in (("pool", pool), ("test", test), ("cal", cal)):
                            x = F[ids].reshape(-1, F.shape[-1]).float() - mu
                            q[part] = (x.half(), (x * x).sum(1), torch.full((len(x),), float("inf"), device=DEVICE))
                        st[n] = (mu, q)
                    taken = np.zeros(len(pool), bool)

                    def add(ks):
                        for n, F in feats.items():
                            mu, q = st[n]
                            new = F[[pool[k] for k in ks]].reshape(-1, F.shape[-1])
                            for part in q:
                                lower(q[part][2], q[part][0], q[part][1], new, mu)
                        taken[ks] = True

                    def fused(part):
                        zs = []
                        for n in feats:
                            q = st[n][1]
                            s = q[part][2].view(-1, per).max(1).values
                            cs = q["cal"][2].view(-1, per).max(1).values
                            zs.append((s - cs.mean()) / cs.std())
                        return torch.stack(zs).mean(0)

                    add(init)
                    for m in sizes:
                        need = m - int(taken.sum())
                        if need > 0:
                            if strat == "alarm":
                                z = fused("pool").cpu().numpy()
                                z[taken] = -np.inf
                                ks = list(np.argsort(-z)[:need])
                            else:
                                ks = [k for k in order if not taken[k]][:need]
                            add(ks)
                        s = fused("test").cpu().numpy()
                        results.append({"cls": c, "fold": fold, "seed": seed, "strategy": strat, "memory": m,
                                        "pool": len(pool), "fraction": m / len(pool), "n_normal": int((y == 0).sum()),
                                        "n_defect": int(y.sum()), "auroc": float(roc_auc_score(y, s)),
                                        "sens95": sens_at_spec(y, s, 0.95), "sens99": sens_at_spec(y, s, 0.99)})
                        scores[f"{c}_{fold}_{seed}_{strat}_{sizes.index(m)}"] = s.astype(np.float32)
                    del st
            last = [r for r in results if r["cls"] == c and r["fold"] == fold and r["memory"] == sizes[3]]
            msg = "  ".join(f"{s}: AUROC {np.mean([r['auroc'] for r in last if r['strategy'] == s]):.3f} "
                            f"S95 {np.mean([r['sens95'] for r in last if r['strategy'] == s]):.3f}" for s in ("alarm", "random"))
            print(f"task {c}/{fold}: pool {len(pool)} calib {len(cal)} test {len(y)} ({y.sum()} defective); "
                  f"at {sizes[3]} crops  {msg}  ({time.time() - t0:.0f}s)", flush=True)
            (out_dir / "results.json").write_text(json.dumps(results))
            np.savez_compressed(out_dir / "scores.npz", **scores)
        lab_dict_path = out_dir / f"labels_{c}.npz"
        np.savez_compressed(lab_dict_path, **{f: lab[[i for i, r in enumerate(crops) if r["fold"] == f]] for f in ("0", "1")})
        del feats
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
