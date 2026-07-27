#!/usr/bin/env python
"""Label-free operating point for the one-class detector via split conformal.

AUROC says how well the detector ranks; it does not give a line a threshold. Here
we set one from normal images only, with a finite-sample distribution-free
guarantee: sort the scores of n calibration normals and take

    tau(alpha) = the ceil((n+1)(1-alpha))-th smallest,

then for exchangeable normal data P(score(X_new) > tau) <= alpha. No anomalies,
no labels, no threshold tuning on the test set. Note alpha is resolution-limited
by the calibration size: alpha < 1/(n+1) admits no finite threshold.

Reported per alpha: the nominal false-alarm rate, the empirical one, and the
recall actually achieved on real defects -- the number an operator needs
("at 1 % false alarms we catch X % of defects").

Two arms separate the guarantee from its assumption:
  deploy   -- calibrate on held-out train normals (disjoint from the memory
              bank), measure the false-alarm rate on the test normals. This is
              the deployable protocol; it holds only if the two are exchangeable.
  exchange -- calibrate and measure on disjoint halves of the held-out train
              normals, where exchangeability holds by construction. If `deploy`
              over-shoots but `exchange` does not, the gap is a distribution
              shift between train and test normals, not a broken guarantee.

Reproducible & argument-free: reads ``configs/conformal.yaml``.

    uv run python scripts/conformal_threshold.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "conformal.yaml"
_cfg = yaml.safe_load(CONFIG_PATH.open())
os.environ["CUDA_VISIBLE_DEVICES"] = str(_cfg["gpu"])

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from run_patchcore import (  # noqa: E402
    batched,
    build_extractor,
    list_images,
    load_image,
    min_dist_to_memory,
)

from semimil.utils import seed_everything  # noqa: E402

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def load_rgb(path: Path, size: int) -> torch.Tensor:
    """MVTec loader, identical to run_mvtec.py (inlined: importing that module
    would re-set CUDA_VISIBLE_DEVICES from its own config)."""
    from PIL import Image

    im = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(im, np.float32) / 255.0).permute(2, 0, 1)
    return (x - MEAN) / STD


# ----------------------------------------------------------------------------- #
# conformal calibration
# ----------------------------------------------------------------------------- #
def conformal_threshold(cal: np.ndarray, alpha: float) -> float | None:
    """Split-conformal threshold guaranteeing a false-alarm rate <= alpha.

    Returns None when alpha is finer than the calibration set can resolve
    (alpha < 1/(n+1)), i.e. when the required order statistic exceeds the sample.
    """
    n = cal.shape[0]
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        return None
    return float(np.sort(cal)[k - 1])


def evaluate_arm(cal_pool, val_normal, abn, alphas, n_draws, n_calib, rng, paired):
    """Repeat conformal calibration over random subsets and average the outcome.

    ``paired`` -> val normals are the complement of the calibration draw inside
    ``cal_pool`` (the exchangeability arm). Otherwise ``val_normal`` is a fixed
    independent set (the deployable arm).
    """
    out = {}
    for alpha in alphas:
        fprs, recalls, taus, n_eff = [], [], [], None
        for _ in range(n_draws):
            perm = rng.permutation(cal_pool.shape[0])
            take = min(n_calib, cal_pool.shape[0] - 1 if paired else cal_pool.shape[0])
            cal = cal_pool[perm[:take]]
            val = cal_pool[perm[take:]] if paired else val_normal
            if val.shape[0] == 0:
                continue
            tau = conformal_threshold(cal, alpha)
            if tau is None:
                n_eff = take
                break
            fprs.append(float((val > tau).mean()))
            recalls.append(float((abn > tau).mean()))
            taus.append(tau)
        if not fprs:
            out[alpha] = {"reportable": False, "n_calib": n_eff,
                          "reason": f"alpha < 1/(n+1) with n={n_eff}"}
            continue
        f = np.array(fprs)
        out[alpha] = {
            "reportable": True,
            "n_calib": int(cal.shape[0]),
            "empirical_fpr_mean": float(f.mean()),
            "empirical_fpr_p90": float(np.quantile(f, 0.90)),
            "coverage_ok_frac": float((f <= alpha).mean()),  # draws honouring the bound
            "recall_mean": float(np.mean(recalls)),
            "recall_std": float(np.std(recalls, ddof=1)) if len(recalls) > 1 else 0.0,
            "tau_mean": float(np.mean(taus)),
            "n_draws": len(fprs),
        }
    return out


# ----------------------------------------------------------------------------- #
# scoring helpers
# ----------------------------------------------------------------------------- #
@torch.no_grad()
def build_bank(paths, extractor, size, bs, ppi, coreset, device, loader):
    bank = []
    for batch in batched(paths, bs):
        x = torch.stack([loader(p, size) for p in batch])
        f = extractor(x)
        f = f.reshape(f.shape[0], -1, f.shape[-1])
        for b in range(f.shape[0]):
            bank.append(f[b, torch.randint(0, f.shape[1], (ppi,))].cpu())
    t = torch.cat(bank, 0)
    if t.shape[0] > coreset:
        t = t[torch.randperm(t.shape[0])[:coreset]]
    return t.to(device).float()


@torch.no_grad()
def score(paths, extractor, size, bs, memory, top_k, device, loader):
    s = []
    for batch in batched(paths, bs):
        x = torch.stack([loader(p, size) for p in batch])
        f = extractor(x)
        d = f.shape[-1]
        for b in range(f.shape[0]):
            dist = min_dist_to_memory(f[b].reshape(-1, d), memory)
            s.append(torch.topk(dist, top_k).values.mean().item())
    return np.array(s)


# ----------------------------------------------------------------------------- #
def run_miic(cfg, device, out_dir):
    m = cfg["miic"]
    raw = Path(m["raw_dir"])
    size, bs = m["model"]["input_size"], m["batch_size"]
    det = m["detector"]
    cache = out_dir / "miic_scores.npz"

    if cache.exists():
        z = np.load(cache)
        s_held, s_tn, s_ta = z["heldout"], z["test_normal"], z["test_abnormal"]
        print(f"[miic] cached scores: {len(s_held)} held-out, "
              f"{len(s_tn)} test normal, {len(s_ta)} test abnormal", flush=True)
    else:
        rng = np.random.default_rng(cfg["seed"])
        normals = list_images(sorted(raw.glob(m["train_normal_glob"])))
        perm = rng.permutation(len(normals))
        bank_paths = [normals[i] for i in perm[: m["n_bank"]]]
        # held-out calibration normals are strictly disjoint from the bank
        held_paths = [normals[i] for i in perm[m["n_bank"] : m["n_bank"] + m["n_heldout"]]]
        test_norm = list_images(sorted(Path(raw, m["test_normal_dir"]).iterdir()))
        test_abn = list_images(sorted(Path(raw, m["test_abnormal_dir"]).iterdir()))
        print(f"[miic] bank={len(bank_paths)} heldout={len(held_paths)} "
              f"test={len(test_norm)}N/{len(test_abn)}A", flush=True)

        extractor = build_extractor(m["model"], device)
        t0 = time.time()
        memory = build_bank(bank_paths, extractor, size, bs, det["patches_per_image"],
                            det["coreset_size"], device, load_image)
        print(f"[miic] bank {tuple(memory.shape)} ({time.time() - t0:.0f}s)", flush=True)
        t0 = time.time()
        s_held = score(held_paths, extractor, size, bs, memory, det["top_k"], device, load_image)
        s_tn = score(test_norm, extractor, size, bs, memory, det["top_k"], device, load_image)
        s_ta = score(test_abn, extractor, size, bs, memory, det["top_k"], device, load_image)
        print(f"[miic] scored ({time.time() - t0:.0f}s)", flush=True)
        np.savez(cache, heldout=s_held, test_normal=s_tn, test_abnormal=s_ta)
        del memory
        torch.cuda.empty_cache()

    rng = np.random.default_rng(cfg["seed"])
    alphas, nd = cfg["alphas"], cfg["n_calib_draws"]
    res = {
        "deploy": evaluate_arm(s_held, s_tn, s_ta, alphas, nd, m["n_calib"], rng, paired=False),
        "exchange": evaluate_arm(s_held, None, s_ta, alphas, nd, m["n_calib"], rng, paired=True),
    }
    res["n"] = {"heldout": len(s_held), "test_normal": len(s_tn),
                "test_abnormal": len(s_ta)}
    return res


def run_mvtec(cfg, device, out_dir):
    mv = cfg["mvtec"]
    root = Path(mv["root"])
    size, bs, det = mv["input_size"], mv["batch_size"], mv["detector"]
    cache = out_dir / "mvtec_scores.json"

    if cache.exists():
        raw_scores = {k: {kk: np.array(vv) for kk, vv in v.items()}
                      for k, v in json.loads(cache.read_text()).items()}
        print(f"[mvtec] cached scores for {len(raw_scores)} categories", flush=True)
    else:
        cats = sorted(p.name for p in root.iterdir() if (p / "train" / "good").is_dir())
        print(f"[mvtec] {len(cats)} categories", flush=True)
        extractor = build_extractor(mv["model"], device)
        rng = np.random.default_rng(cfg["seed"])
        raw_scores = {}
        for cat in cats:
            t0 = time.time()
            good = sorted((root / cat / "train" / "good").glob("*.png"))
            perm = rng.permutation(len(good))
            n_bank = round(mv["bank_frac"] * len(good))
            bank_paths = [good[i] for i in perm[:n_bank]]
            cal_paths = [good[i] for i in perm[n_bank:]]
            memory = build_bank(bank_paths, extractor, size, bs, det["patches_per_image"],
                                det["coreset_size"], device, load_rgb)
            test_good, test_bad = [], []
            for defect in sorted((root / cat / "test").iterdir()):
                ps = sorted(defect.glob("*.png"))
                (test_good if defect.name == "good" else test_bad).extend(ps)
            raw_scores[cat] = {
                "cal": score(cal_paths, extractor, size, bs, memory, det["top_k"],
                             device, load_rgb),
                "test_normal": score(test_good, extractor, size, bs, memory, det["top_k"],
                                     device, load_rgb),
                "test_abnormal": score(test_bad, extractor, size, bs, memory, det["top_k"],
                                       device, load_rgb),
            }
            print(f"  {cat:<12} cal={len(cal_paths):>3} test={len(test_good)}N/"
                  f"{len(test_bad)}A ({time.time() - t0:.0f}s)", flush=True)
            del memory
            torch.cuda.empty_cache()
        cache.write_text(json.dumps(
            {k: {kk: vv.tolist() for kk, vv in v.items()} for k, v in raw_scores.items()},
            indent=2))

    rng = np.random.default_rng(cfg["seed"])
    per_cat = {}
    for cat, s in raw_scores.items():
        # train/good is small, so each draw takes a fraction of the available
        # calibration images: using all of them every time would give one fixed
        # threshold and no read on how much it moves with the calibration draw.
        n_cal = max(1, round(mv["calib_frac"] * len(s["cal"])))
        per_cat[cat] = evaluate_arm(s["cal"], s["test_normal"], s["test_abnormal"],
                                    mv["alphas"], cfg["n_calib_draws"],
                                    n_cal, rng, paired=False)
        per_cat[cat]["n_calib_available"] = len(s["cal"])
    return per_cat


def main() -> None:
    cfg = _cfg
    seed_everything(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device}  out={out_dir}", flush=True)

    summary: dict = {"config": cfg, "miic": run_miic(cfg, device, out_dir)}
    if cfg["mvtec"]["enabled"]:
        summary["mvtec"] = run_mvtec(cfg, device, out_dir)
    (out_dir / "results.json").write_text(json.dumps(summary, indent=2, default=str))

    # ---- report --------------------------------------------------------------
    print("\n=== MIIC: conformal operating point (mean over "
          f"{cfg['n_calib_draws']} calibration draws) ===")
    hdr = f"{'alpha':>7}{'emp. FPR':>11}{'p90 FPR':>10}{'bound held':>12}{'recall':>10}"
    for arm in ("deploy", "exchange"):
        print(f"-- {arm}")
        print(hdr)
        for a in cfg["alphas"]:
            r = summary["miic"][arm][a]
            if not r["reportable"]:
                print(f"{a:>7}   not reportable ({r['reason']})")
                continue
            print(f"{a:>7}{r['empirical_fpr_mean']:>11.4f}{r['empirical_fpr_p90']:>10.4f}"
                  f"{r['coverage_ok_frac']:>11.0%}  {r['recall_mean']:>9.3f}")

    if cfg["mvtec"]["enabled"]:
        print("\n=== MVTec-AD: conformal validity per category (deployable arm) ===")
        print(f"{'category':<12}{'n_cal':>6}" +
              "".join(f"{'a=' + str(a):>18}" for a in cfg["mvtec"]["alphas"]))
        print(f"{'':12}{'':6}" + "".join(f"{'FPR / recall':>18}" for _ in cfg["mvtec"]["alphas"]))
        rows = summary["mvtec"]
        for cat in sorted(rows):
            line = f"{cat:<12}{rows[cat]['n_calib_available']:>6}"
            for a in cfg["mvtec"]["alphas"]:
                r = rows[cat][a]
                line += (f"{r['empirical_fpr_mean']:>8.3f} /{r['recall_mean']:>8.3f}"
                         if r["reportable"] else f"{'--':>18}")
            print(line)
        for a in cfg["mvtec"]["alphas"]:
            ok = [rows[c][a] for c in rows if rows[c][a]["reportable"]]
            if ok:
                print(f"  mean over {len(ok)} categories at alpha={a}: "
                      f"FPR={np.mean([r['empirical_fpr_mean'] for r in ok]):.4f}  "
                      f"recall={np.mean([r['recall_mean'] for r in ok]):.3f}  "
                      f"bound held in {np.mean([r['coverage_ok_frac'] for r in ok]):.0%} of draws")

    print(f"\nsaved -> {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
