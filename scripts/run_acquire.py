#!/usr/bin/env python
"""Alarm-driven acquisition of normals: the detector's own false alarms choose what to add to its memory.

Premise (diagnosis of the paper): at a 1 % false-alarm rate the threshold is set by rare but
normal structures that the memory does not cover.  The images that the detector flags and an
inspector confirms as normal are therefore exactly the ones the memory lacks.  This script
simulates that closed loop as pool-based active acquisition of *normals*, with no defect labels:

  memory <- all tokens of n_initial random training normals
  repeat for each round of size B:
      score every remaining pool image with the current memory (image score = max patch
      nearest-neighbor distance; with two representations, the mean of their calibration
      z-scores, i.e. the proposed detector)
      alarm  : add the B highest-scoring images (confirmed normal, since the pool is defect-free)
      random : add the next B images of the random order (= the nested banks of run_bigbank.py)
      record the test and calibration maps -> raw/<rep>_n<M>.npz, metrics -> results.json

Nearest-neighbor distances are maintained incrementally: adding tokens can only lower them, so
each round costs (pool tokens) x (new tokens).  Pool tokens stay in CPU RAM (fp16, ~25 GB per
representation) and are streamed through the GPU; test/calibration distances use the precise
path of run_bigbank.py, pool distances a fused fp16 GEMM (checked against fp32 cdist with
``check: true``).  Argument-free: configs/acquire_alarm.yaml (or SEMIMIL_CONFIG).
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
from run_patchcore import build_extractor, list_images, load_image  # noqa: E402

from semimil.utils import seed_everything  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(os.environ.get("SEMIMIL_CONFIG", REPO_ROOT / "configs" / "acquire_alarm.yaml"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = True


@torch.no_grad()
def extract_into(paths, extractor, size, bs, out, per_img):
    """Fill the preallocated CPU tensor out[len(paths) * per_img, D] in place (no cat peak)."""
    for i in range(0, len(paths), bs):
        x = torch.stack([load_image(p, size) for p in paths[i : i + bs]])
        f = extractor(x).reshape(-1, out.shape[1]).half().cpu()
        out[i * per_img : i * per_img + len(f)] = f


class Rep:
    """One token representation: pool tokens on the CPU, queries and running NN distances on the GPU."""

    def __init__(self, name, cfg_model, pool_paths, q_paths, size, bs, n_initial, keep_pool):
        self.name = name
        extractor = build_extractor(cfg_model, DEVICE)
        f = extractor(torch.stack([load_image(pool_paths[0], size)]))
        self.per, self.dim = f.reshape(-1, f.shape[-1]).shape
        t0 = time.time()
        self.pool = torch.empty((len(pool_paths) * self.per, self.dim), dtype=torch.float16)
        extract_into(pool_paths, extractor, size, bs, self.pool, self.per)
        q = torch.empty((len(q_paths) * self.per, self.dim), dtype=torch.float16)
        extract_into(q_paths, extractor, size, bs, q, self.per)
        del extractor
        torch.cuda.empty_cache()
        # center on the initial memory (a global shift keeps distances, conditions the expansion)
        self.mu = self.pool[: n_initial * self.per].float().mean(0).to(DEVICE)
        self.q = torch.empty(q.shape, dtype=torch.float16, device=DEVICE)
        for i in range(0, len(q), 500_000):
            self.q[i : i + 500_000] = (q[i : i + 500_000].to(DEVICE).float() - self.mu).half()
        self.q2 = torch.cat([(self.q[i : i + 500_000].float() ** 2).sum(1) for i in range(0, len(q), 500_000)])
        self.dmin_q = torch.full((len(q),), float("inf"), device=DEVICE)
        self.dmin_pool = torch.full((len(self.pool),), float("inf"), device=DEVICE) if keep_pool else None
        print(f"[{name}] pool {tuple(self.pool.shape)} queries {tuple(q.shape)} "
              f"({self.per} tokens/image, {time.time() - t0:.0f}s)", flush=True)

    @torch.no_grad()
    def add_images(self, idx, q_chunk, p_chunk, b_chunk, profile=False):
        """Lower every running NN distance with the tokens of the pool images idx."""
        idx = torch.as_tensor(idx, dtype=torch.long)
        T = self.pool.view(-1, self.per, self.dim)[idx].reshape(-1, self.dim)
        T = (T.to(DEVICE).float() - self.mu).half()
        t2 = (T.float() ** 2).sum(1)
        t0 = time.time()
        for qs in range(0, len(self.q), q_chunk):  # precise path (fp32 expansion), as run_bigbank
            q, q2 = self.q[qs : qs + q_chunk], self.q2[qs : qs + q_chunk, None]
            best = self.dmin_q[qs : qs + q_chunk]
            for bs in range(0, len(T), b_chunk):
                d2 = (q @ T[bs : bs + b_chunk].T).float()
                d2.mul_(-2.0).add_(q2).add_(t2[None, bs : bs + b_chunk])
                best = torch.minimum(best, torch.amin(d2, 1).clamp_min_(0).sqrt_())
                del d2
            self.dmin_q[qs : qs + q_chunk] = best
        if self.dmin_pool is None:
            return
        torch.cuda.synchronize()
        t1 = time.time()
        # fast path: one fp16 GEMM on augmented vectors gives t2 - 2 p.t directly; min over t; + p2 in fp32
        Taug = torch.cat([(-2.0 * T.float()).half(), t2.half()[:, None]], 1)
        if float(t2.max()) > 3e4:
            print(f"[{self.name}] warning: max |t|^2 = {float(t2.max()):.0f}, near the fp16 range", flush=True)
        ones = torch.ones((p_chunk, 1), dtype=torch.float16, device=DEVICE)
        for ps in range(0, len(self.pool), p_chunk):
            p = self.pool[ps : ps + p_chunk].to(DEVICE).float() - self.mu
            p2 = (p * p).sum(1)
            paug = torch.cat([p.half(), ones[: len(p)]], 1)
            best = torch.full((len(p),), float("inf"), device=DEVICE)
            for bs in range(0, len(T), b_chunk):
                best = torch.minimum(best, torch.amin(paug @ Taug[bs : bs + b_chunk].T, 1).float())
            self.dmin_pool[ps : ps + p_chunk] = torch.minimum(self.dmin_pool[ps : ps + p_chunk],
                                                               (best + p2).clamp_min_(0).sqrt_())
        if profile:
            torch.cuda.synchronize()
            pairs = len(self.pool) * len(T)
            print(f"[{self.name}] +{len(idx)} images: queries {t1 - t0:.1f}s, pool {time.time() - t1:.1f}s "
                  f"({pairs / (time.time() - t1) / 1e9:.1f} G pairs/s)", flush=True)

    def image_scores(self, n_images):
        return self.dmin_pool.view(n_images, self.per).max(1).values

    def query_scores(self):
        return self.dmin_q.view(-1, self.per).max(1).values

    def check(self, n_initial, n=2048):
        """Max error of the fast pool path against fp32 cdist, on n pool tokens outside the memory."""
        p = self.pool[-n:].to(DEVICE).float() - self.mu
        ref = torch.cdist(p, self.pool[: n_initial * self.per].to(DEVICE).float() - self.mu).min(1).values
        err = (self.dmin_pool[-n:] - ref).abs()
        print(f"[{self.name}] fast path vs cdist: max err {err.max():.4f} mean {err.mean():.5f} "
              f"(mean dist {ref.mean():.3f})", flush=True)


def metrics(score, n_norm, areas, calib):
    normal, abn = score[:n_norm], score[n_norm:]
    y = np.r_[np.zeros(n_norm), np.ones(len(abn))]
    thr = float(np.percentile(normal, 99))
    thr_c = float(np.percentile(calib, 99))
    small, medium = areas < 0.1, (areas >= 0.1) & (areas < 1)
    return {"auroc": float(roc_auc_score(y, score)), "threshold": thr,
            "sens_spec99": float((abn >= thr).mean()), "sens_small": float((abn[small] >= thr).mean()),
            "sens_medium": float((abn[medium] >= thr).mean()),
            "auroc_small": float(roc_auc_score(np.r_[np.zeros(n_norm), np.ones(small.sum())], np.r_[normal, abn[small]])),
            "calib_threshold": thr_c, "calib_fpr": float((normal >= thr_c).mean()),
            "calib_sens": float((abn >= thr_c).mean()), "calib_sens_small": float((abn[small] >= thr_c).mean())}


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
    gen = torch.Generator().manual_seed(cfg["seed"])  # same calibration split as run_bigbank
    perm = torch.randperm(len(train), generator=gen).tolist()
    n_cal = cfg["calibration"]["n_images"]
    calib, pool = [train[j] for j in perm[:n_cal]], [train[j] for j in perm[n_cal:]]
    order = torch.randperm(len(pool), generator=torch.Generator().manual_seed(cfg["bank_seed"])).tolist()
    pool = [pool[j] for j in order]  # the random strategy walks this order: prefixes = run_bigbank banks
    if cfg.get("pool_images"):  # smoke test only
        pool = pool[: cfg["pool_images"]]
    strategy, n0, rounds = cfg["strategy"], cfg["n_initial"], list(cfg["rounds"])
    if strategy == "random":  # only the prefix that will ever be added is needed
        pool = pool[: n0 + sum(rounds)]
    n_pool, n_t, n_norm = len(pool), len(test_norm) + len(test_abn), len(test_norm)
    size, bs = cfg["input_size"], cfg["batch_size"]
    qc, pc, bc = cfg["q_chunk"], cfg["p_chunk"], cfg["b_chunk"]
    print(f"device={DEVICE} strategy={strategy} pool={n_pool} initial={n0} rounds={rounds} "
          f"calib={len(calib)} test={n_norm}N/{len(test_abn)}A", flush=True)

    reps = [Rep(r["name"], r["model"], pool, test_norm + test_abn + calib, size, bs, n0, strategy == "alarm")
            for r in cfg["representations"]]
    areas = np.array([(np.asarray(Image.open(mask_dir / f"{p.stem}_mask.jpg").convert("L")) > 127).mean() * 100
                      for p in test_abn])
    acquired = list(range(n0))
    taken = torch.zeros(n_pool, dtype=torch.bool, device=DEVICE)
    taken[:n0] = True
    t0 = time.time()
    prof = bool(cfg.get("profile"))
    state_path = out_dir / "state.pt"
    results, log, done = [], [], 0
    if cfg.get("resume") and state_path.exists():  # continue a job that hit the time limit
        st = torch.load(state_path)
        acquired, done = st["acquired"], st["round"]
        results = json.loads((out_dir / "results.json").read_text())
        log = json.loads((out_dir / "acquired.json").read_text())
        for rep in reps:
            rep.dmin_q = st["dmin_q"][rep.name].to(DEVICE)
            if rep.dmin_pool is not None:
                rep.dmin_pool = st["dmin_pool"][rep.name].to(DEVICE)
        taken[torch.as_tensor(acquired, device=DEVICE)] = True
        print(f"resumed after round {done}: memory of {len(acquired)} images", flush=True)
    else:
        for rep in reps:
            rep.add_images(acquired, qc, pc, bc, prof)
        print(f"initial memory of {n0} images ({time.time() - t0:.0f}s)", flush=True)
        if cfg.get("check") and strategy == "alarm":  # random runs keep no pool distances
            for rep in reps:
                rep.check(n0)

    def save_state(round_id):
        torch.save({"round": round_id, "acquired": acquired,
                    "dmin_q": {r.name: r.dmin_q.cpu() for r in reps},
                    "dmin_pool": {r.name: (r.dmin_pool.cpu() if r.dmin_pool is not None else None) for r in reps}},
                   state_path)

    def checkpoint(round_id, picked, pick_scores):
        m_imgs = len(acquired)
        entry = {"round": round_id, "memory_images": m_imgs, "memory_tokens": m_imgs * reps[0].per, "reps": {}}
        zs = []
        for rep in reps:
            s = rep.query_scores().cpu().numpy()
            test, cal = s[:n_t], s[n_t:]
            d = rep.dmin_q.view(-1, rep.per).cpu()
            g = int(round(rep.per ** 0.5))
            np.savez_compressed(out_dir / "raw" / f"{rep.name}_n{m_imgs}.npz",
                                test_map=d[:n_t].reshape(n_t, g, g).half().numpy(),
                                calib_map=d[n_t:].reshape(-1, g, g).half().numpy())
            entry["reps"][rep.name] = metrics(test, n_norm, areas, cal)
            zs.append((s - cal.mean()) / cal.std())
        if len(reps) > 1:
            z = np.mean(zs, 0)
            entry["fused"] = metrics(z[:n_t], n_norm, areas, z[n_t:])
        results.append(entry)
        log.append({"round": round_id, "added": [int(i) for i in picked],
                    "scores": [float(v) for v in pick_scores]})
        (out_dir / "results.json").write_text(json.dumps(results, indent=1))
        (out_dir / "acquired.json").write_text(json.dumps(log))
        head = entry.get("fused") or entry["reps"][reps[0].name]
        print(f"memory={m_imgs:6d} imgs  AUROC {head['auroc']:.4f}  Sens@99 {head['sens_spec99']:.3f}  "
              f"small {head['sens_small']:.3f}  thr {head['threshold']:.3f}  calibFPR {head['calib_fpr']:.4f}  "
              f"({time.time() - t0:.0f}s)", flush=True)

    if done == 0:
        checkpoint(0, acquired, [])
    for ri, B in enumerate(rounds, 1):
        if ri <= done:
            continue
        B = min(B, n_pool - len(acquired))
        if B <= 0:
            break
        if strategy == "alarm":
            zs = []
            for rep in reps:
                s = rep.image_scores(n_pool)
                cal = rep.query_scores()[n_t:]
                zs.append((s - cal.mean()) / cal.std())
            z = torch.stack(zs).mean(0)
            z[taken] = float("-inf")
            sc, picked = torch.topk(z, B)
            picked, sc = picked.cpu().tolist(), sc.cpu().tolist()
        else:
            picked, sc = list(range(len(acquired), len(acquired) + B)), []
        for rep in reps:
            rep.add_images(picked, qc, pc, bc, prof)
        taken[picked] = True
        acquired.extend(picked)
        checkpoint(ri, picked, sc)
        save_state(ri)


if __name__ == "__main__":
    main()
