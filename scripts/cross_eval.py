#!/usr/bin/env python
"""Cross-evaluate Track A (one-class MIL) and Track B (attention-MIL) on BOTH
the real GT test and the held-out synthetic test.

Track A = unsupervised MIL: memory bank of normal patches + top-k pooling of
per-instance distances (no labels). Track B = supervised MIL: gated-attention
head trained on synthetic positives. Same DINOv2 instances feed both, so the
2x2 (model x domain) table isolates the synthetic->real domain gap.

Reproducible & argument-free: reads ``configs/cross_eval.yaml``.

    uv run python scripts/cross_eval.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_patchcore import (
    build_extractor,
    list_images,
    min_dist_to_memory,
)
from train_mil import GatedAttentionMIL, precompute, predict

from semimil.utils import seed_everything

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "cross_eval.yaml"


def build_memory(paths, extractor, size, bs, cfg, gen, device):
    feats, _ = precompute(paths, extractor, size, bs)  # [n, N, D] fp16
    ppi = cfg["memory"]["patches_per_image"]
    picks = [
        feats[b, torch.randint(0, feats.shape[1], (ppi,), generator=gen)]
        for b in range(len(feats))
    ]
    bank = torch.cat(picks, dim=0)
    if bank.shape[0] > cfg["memory"]["coreset_size"]:
        sel = torch.randperm(bank.shape[0], generator=gen)[: cfg["memory"]["coreset_size"]]
        bank = bank[sel]
    return bank.to(device).float()


def track_a_scores(bags, memory, top_k, device):
    """Unsupervised-MIL image scores: top-k mean of per-instance NN distances."""
    scores = []
    for i in range(bags.shape[0]):
        d = min_dist_to_memory(bags[i].to(device).float(), memory)
        scores.append(torch.topk(d, top_k).values.mean().item())
    return np.array(scores)


def metrics(y, s, pct=99):
    """Threshold-free (AUROC/AUPRC) + threshold-based (balanced acc, kappa).

    Operating point = ``pct``-percentile of the *normal* scores in this domain
    (i.e. ~ (100-pct)% FPR). balanced accuracy and Cohen's kappa need a threshold;
    this fixes one consistently across every experiment.
    """
    thr = float(np.percentile(s[y == 0], pct))
    pred = (s >= thr).astype(int)
    return {
        "auroc": float(roc_auc_score(y, s)),
        "auprc": float(average_precision_score(y, s)),
        "balanced_acc": float(balanced_accuracy_score(y, pred)),
        "cohen_kappa": float(cohen_kappa_score(y, pred)),
        "threshold_pct": pct,
    }


def main() -> None:
    cfg = yaml.safe_load(CONFIG_PATH.open())
    seed_everything(cfg["seed"])
    gen = torch.Generator().manual_seed(cfg["seed"])
    device = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if cfg["device"] == "auto"
        else cfg["device"]
    )
    raw = Path(cfg["data"]["raw_dir"])
    size, bs = cfg["model"]["input_size"], cfg["model"]["batch_size"]
    out_dir = Path(cfg["output"]["dir"]) / cfg["run_name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device}  out={out_dir}")
    extractor = build_extractor(cfg["model"], device)

    # ---- paths ----
    d = cfg["data"]
    train_normal = list_images(sorted(raw.glob(d["train_normal_glob"])))
    train_normal = [train_normal[i] for i in torch.randperm(len(train_normal), generator=gen)]
    pos = list_images(sorted(raw.glob(d["pos_glob"])))
    neg = list_images(sorted(raw.glob(d["neg_glob"])))
    neg = neg + train_normal[: d["n_extra_normal"]]
    mem_imgs = train_normal[: cfg["memory"]["n_images"]]

    real_norm = list_images(sorted(Path(raw, d["test_normal_dir"]).iterdir()))
    real_abn = list_images(sorted(Path(raw, d["test_abnormal_dir"]).iterdir()))
    syn_norm = list_images(sorted(raw.glob(d["syn_test_neg_glob"])))
    syn_abn = list_images(sorted(raw.glob(d["syn_test_pos_glob"])))
    print(f"real: {len(real_norm)}N/{len(real_abn)}A   synthetic: {len(syn_norm)}N/{len(syn_abn)}A")

    # ---- Track A: memory bank ----
    print("[A] building memory bank ...")
    memory = build_memory(mem_imgs, extractor, size, bs, cfg, gen, device)

    # ---- Track B: train attention-MIL ----
    print("[B] precomputing train features + training ...")
    Fpos, _ = precompute(pos, extractor, size, bs)
    Fneg, _ = precompute(neg, extractor, size, bs)
    X = torch.cat([Fpos, Fneg])
    y = torch.cat([torch.ones(len(Fpos)), torch.zeros(len(Fneg))])
    perm = torch.randperm(len(X), generator=gen)
    X, y = X[perm], y[perm]
    dim = X.shape[-1]
    mc = cfg["mil"]
    model = GatedAttentionMIL(dim, mc["hidden_dim"], mc["attn_dim"], mc["dropout"]).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"]
    )
    pw = torch.tensor([(y == 0).sum() / (y == 1).sum().clamp(min=1)], device=device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
    bbs = cfg["train"]["bag_batch_size"]
    for _ in range(cfg["train"]["epochs"]):
        model.train()
        idx = torch.randperm(len(X), generator=gen)
        for i in range(0, len(idx), bbs):
            sel = idx[i : i + bbs]
            logit, _ = model(X[sel].to(device).float())
            loss = loss_fn(logit, y[sel].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()

    # ---- score each eval domain with both models (shared features) ----
    results = {}
    for domain, negs, abns in [
        ("real_GT", real_norm, real_abn),
        ("synthetic", syn_norm, syn_abn),
    ]:
        Fn, _ = precompute(negs, extractor, size, bs)
        Fa, _ = precompute(abns, extractor, size, bs)
        F_all = torch.cat([Fn, Fa])
        y_all = np.array([0] * len(Fn) + [1] * len(Fa))
        sa = track_a_scores(F_all, memory, cfg["score"]["top_k"], device)
        sb, _ = predict(model, F_all, device)
        results[domain] = {
            "n_normal": len(Fn),
            "n_abnormal": len(Fa),
            "track_A_oneclass": metrics(y_all, sa),
            "track_B_attnMIL": metrics(y_all, sb),
        }
        names = [p.name for p in list(negs) + list(abns)]
        with (out_dir / f"scores_{domain}.csv").open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["name", "label", "score_A", "score_B"])
            for nm, yl, xa, xb in zip(names, y_all, sa, sb, strict=True):
                w.writerow([nm, int(yl), float(xa), float(xb)])

    payload = {"config": cfg, "results": results}
    (out_dir / "cross_metrics.json").write_text(json.dumps(payload, indent=2))

    # ---- table ----
    print("\n=== CROSS-EVAL  (bal_acc / AUPRC / kappa @ 1% FPR operating point) ===")
    hdr = f"{'model':<24}{'domain':<12}{'bal_acc':>9}{'AUPRC':>9}{'kappa':>9}{'AUROC':>9}"
    print(hdr)
    rows = [("Track A one-class", "track_A_oneclass"), ("Track B attn-MIL", "track_B_attnMIL")]
    for domain, r in results.items():
        for label, key in rows:
            m = r[key]
            print(
                f"{label:<24}{domain:<12}"
                f"{m['balanced_acc']:>9.3f}{m['auprc']:>9.3f}"
                f"{m['cohen_kappa']:>9.3f}{m['auroc']:>9.3f}"
            )
    print(f"\nsaved -> {out_dir}")


if __name__ == "__main__":
    main()
