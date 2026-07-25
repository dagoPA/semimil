#!/usr/bin/env python
"""Track B -- weakly-supervised attention-MIL anomaly detection on MIIC.

Each image is a *bag* of feature-map-cell *instances* (shared frozen extractor
with Track A). A gated-attention MIL head (Ilse et al., 2018) is trained to
classify bags normal/abnormal using SYNTHETIC positives (the Inpainting split);
the learned attention over instances gives localization. Evaluation is on the
REAL abnormal test set, which the model never sees in training.

Features are precomputed once (frozen backbone) and the small MIL head trains
many epochs on top -- the standard efficient MIL pattern.

Reproducible & argument-free: all settings come from ``configs/mil.yaml``.

    uv run python scripts/train_mil.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from scipy.ndimage import gaussian_filter
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_patchcore import batched, build_extractor, list_images, load_image, load_mask

from semimil.utils import seed_everything

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "mil.yaml"


# ----------------------------------------------------------------------------- #
# features
# ----------------------------------------------------------------------------- #
@torch.no_grad()
def precompute(paths, extractor, size, bs) -> tuple[torch.Tensor, int]:
    """Extract per-instance features for a list of images -> ([n, N, D] fp16, grid)."""
    out, grid = [], None
    for batch in batched(paths, bs):
        x = torch.stack([load_image(p, size) for p in batch])
        feats = extractor(x)  # [B, Hf, Wf, D]
        grid = feats.shape[1]
        out.append(feats.reshape(feats.shape[0], -1, feats.shape[-1]).half().cpu())
    return torch.cat(out, dim=0), grid


# ----------------------------------------------------------------------------- #
# model
# ----------------------------------------------------------------------------- #
class GatedAttentionMIL(nn.Module):
    """Gated-attention MIL head over instance features (Ilse et al., 2018)."""

    def __init__(self, in_dim: int, hidden: int, attn_dim: int, dropout: float):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout))
        self.attn_v = nn.Linear(hidden, attn_dim)
        self.attn_u = nn.Linear(hidden, attn_dim)
        self.attn_w = nn.Linear(attn_dim, 1)
        self.classifier = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """[B, N, D] -> (bag logits [B], attention [B, N])."""
        h = self.proj(x)  # [B, N, hidden]
        gate = torch.tanh(self.attn_v(h)) * torch.sigmoid(self.attn_u(h))
        a = torch.softmax(self.attn_w(gate).squeeze(-1), dim=1)  # [B, N]
        z = torch.einsum("bn,bnh->bh", a, h)  # attention-weighted bag embedding
        return self.classifier(z).squeeze(-1), a


# ----------------------------------------------------------------------------- #
# eval helpers
# ----------------------------------------------------------------------------- #
@torch.no_grad()
def predict(model, feats, device, bs=64):
    """Return (probs [n], attention [n, N]) for a bag feature tensor."""
    model.eval()
    probs, attn = [], []
    for i in range(0, feats.shape[0], bs):
        x = feats[i : i + bs].to(device).float()
        logit, a = model(x)
        probs.append(torch.sigmoid(logit).cpu())
        attn.append(a.cpu())
    return torch.cat(probs).numpy(), torch.cat(attn).numpy()


def main() -> None:
    cfg = yaml.safe_load(CONFIG_PATH.open())
    seed_everything(cfg["seed"])
    device = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if cfg["device"] == "auto"
        else cfg["device"]
    )
    raw = Path(cfg["data"]["raw_dir"])
    size = cfg["model"]["input_size"]
    bs = cfg["model"]["batch_size"]
    out_dir = Path(cfg["output"]["dir"]) / cfg["run_name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device}  run={cfg['run_name']}  out={out_dir}")

    extractor = build_extractor(cfg["model"], device)
    g = torch.Generator().manual_seed(cfg["seed"])

    # --- gather paths ---------------------------------------------------------
    pos = list_images(sorted(raw.glob(cfg["data"]["pos_glob"])))
    neg = list_images(sorted(raw.glob(cfg["data"]["neg_glob"])))
    extra = list_images(sorted(raw.glob(cfg["data"]["extra_normal_glob"])))
    keep = torch.randperm(len(extra), generator=g)[: cfg["data"]["n_extra_normal"]]
    neg = neg + [extra[i] for i in keep]
    test_norm = list_images(sorted(Path(raw, cfg["data"]["test_normal_dir"]).iterdir()))
    test_abn = list_images(sorted(Path(raw, cfg["data"]["test_abnormal_dir"]).iterdir()))
    print(f"pos={len(pos)}  neg={len(neg)}  test_normal={len(test_norm)}  test_abn={len(test_abn)}")

    # --- precompute features (once) -------------------------------------------
    print("[features] extracting ...")
    Fpos, grid = precompute(pos, extractor, size, bs)
    Fneg, _ = precompute(neg, extractor, size, bs)
    Ftn, _ = precompute(test_norm, extractor, size, bs)
    Fta, _ = precompute(test_abn, extractor, size, bs)
    d = Fpos.shape[-1]
    print(f"[features] grid={grid}x{grid}  dim={d}")

    # --- train/val split of synthetic bags ------------------------------------
    X = torch.cat([Fpos, Fneg], dim=0)
    y = torch.cat([torch.ones(len(Fpos)), torch.zeros(len(Fneg))])
    perm = torch.randperm(len(X), generator=g)
    X, y = X[perm], y[perm]
    n_val = int(cfg["train"]["val_frac"] * len(X))
    Xval, yval = X[:n_val], y[:n_val]
    Xtr, ytr = X[n_val:], y[n_val:]

    # --- model / optim --------------------------------------------------------
    mc = cfg["mil"]
    model = GatedAttentionMIL(d, mc["hidden_dim"], mc["attn_dim"], mc["dropout"]).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"]
    )
    pos_weight = torch.tensor([(ytr == 0).sum() / (ytr == 1).sum().clamp(min=1)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    bbs = cfg["train"]["bag_batch_size"]

    best_auroc, best_state = -1.0, None
    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        idx = torch.randperm(len(Xtr), generator=g)
        tot = 0.0
        for i in range(0, len(idx), bbs):
            sel = idx[i : i + bbs]
            xb = Xtr[sel].to(device).float()
            yb = ytr[sel].to(device)
            logit, _ = model(xb)
            loss = loss_fn(logit, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(sel)
        vprob, _ = predict(model, Xval, device)
        vauroc = roc_auc_score(yval.numpy(), vprob)
        if vauroc > best_auroc:
            best_auroc = vauroc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 5 == 0 or epoch == cfg["train"]["epochs"] - 1:
            print(f"  epoch {epoch:3d}  loss {tot / len(Xtr):.4f}  val_auroc {vauroc:.4f}")

    model.load_state_dict(best_state)
    print(f"[train] best synthetic-val AUROC = {best_auroc:.4f}")

    # --- evaluate on REAL test ------------------------------------------------
    pn, _ = predict(model, Ftn, device)
    pa, attn_a = predict(model, Fta, device)
    y_img = np.array([0] * len(pn) + [1] * len(pa))
    s_img = np.concatenate([pn, pa])

    # localization from attention (abnormal bags) vs masks, at map_size
    ms = cfg["eval"]["map_size"]
    px_pred, px_true = [], []
    amaps = []
    for k, p in enumerate(test_abn):
        amap = attn_a[k].reshape(grid, grid)
        up = F.interpolate(
            torch.tensor(amap)[None, None], size=(ms, ms), mode="bilinear", align_corners=False
        )[0, 0].numpy()
        amaps.append(up)
        px_pred.append(up.reshape(-1))
        mpath = Path(raw, cfg["data"]["test_mask_dir"], f"{p.stem}_mask.jpg")
        px_true.append(load_mask(mpath, ms).reshape(-1))
    px_pred = np.concatenate(px_pred)
    px_true = np.concatenate(px_true)

    metrics = {
        "run_name": cfg["run_name"],
        "model_type": cfg["model"].get("type", "cnn"),
        "synthetic_val_auroc": float(best_auroc),
        "image_auroc": float(roc_auc_score(y_img, s_img)),
        "image_auprc": float(average_precision_score(y_img, s_img)),
        "pixel_auroc": float(roc_auc_score(px_true, px_pred)),
        "pixel_auprc": float(average_precision_score(px_true, px_pred)),
        "n_pos": len(pos),
        "n_neg": len(neg),
        "grid": grid,
        "config": cfg,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    with (out_dir / "image_scores.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "label", "score"])
        for p, s in zip(test_norm, pn, strict=True):
            w.writerow([p.name, 0, float(s)])
        for p, s in zip(test_abn, pa, strict=True):
            w.writerow([p.name, 1, float(s)])

    save_attn_viz(out_dir, test_abn, pa, amaps, cfg, size, raw)

    print("\n=== TRACK B RESULTS (real test) ===")
    for k in ("image_auroc", "image_auprc", "pixel_auroc", "pixel_auprc"):
        print(f"  {k:14s}: {metrics[k]:.4f}")
    print(f"\nsaved -> {out_dir}")


def save_attn_viz(out_dir, abn_paths, probs, amaps, cfg, size, raw) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = np.argsort(probs)
    n = cfg["eval"]["n_viz"]
    picks = list(order[-(n // 2) :][::-1]) + list(order[: n - n // 2])
    mask_dir = Path(raw, cfg["data"]["test_mask_dir"])
    fig, axes = plt.subplots(len(picks), 4, figsize=(12, 3 * len(picks)))
    for j, c in enumerate(["original", "attention", "overlay", "GT mask"]):
        axes[0, j].set_title(c, fontsize=12)
    for i, k in enumerate(picks):
        p = abn_paths[k]
        img = np.asarray(Image.open(p).convert("L").resize((size, size)))
        heat = gaussian_filter(
            np.asarray(Image.fromarray(amaps[k]).resize((size, size), Image.BILINEAR)),
            cfg["eval"].get("sigma", 4.0),
        )
        heat = (heat - heat.min()) / (np.ptp(heat) + 1e-8)
        mask = load_mask(mask_dir / f"{p.stem}_mask.jpg", size) * 255
        axes[i, 0].imshow(img, cmap="gray")
        axes[i, 1].imshow(heat, cmap="jet")
        axes[i, 2].imshow(img, cmap="gray")
        axes[i, 2].imshow(heat, cmap="jet", alpha=0.5)
        axes[i, 3].imshow(mask, cmap="gray")
        axes[i, 0].set_ylabel(f"{p.name}\nprob={probs[k]:.2f}", fontsize=9)
        for j in range(4):
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
    fig.suptitle(f"Attention-MIL [{cfg['run_name']}] — attention vs GT (real abnormals)", y=1.0)
    fig.tight_layout()
    fig.savefig(out_dir / "attention_examples.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
