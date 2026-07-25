#!/usr/bin/env python
"""DRAEM baseline (Zavrtanik et al., ICCV 2021) on MIIC.

A reconstructive U-Net learns to restore normal images from synthetically
corrupted inputs; a discriminative U-Net segments the anomaly from the pair
(input, reconstruction). Trained only on normal images with synthetic anomalies
(a multi-octave noise mask blended with texture from another normal image, a
stand-in for the DTD source used in the paper). Test score = max of the smoothed
anomaly map.

Reproducible & argument-free: reads ``configs/draem.yaml``.
    CUDA_VISIBLE_DEVICES=2 uv run python scripts/run_draem.py
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "draem.yaml"
_cfg = yaml.safe_load(CONFIG_PATH.open())
os.environ["CUDA_VISIBLE_DEVICES"] = str(_cfg["gpu"])

import csv  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from semimil.utils import seed_everything  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------- #
# synthetic anomaly
# --------------------------------------------------------------------------- #
def blob_mask(size, rng):
    """Multi-octave upsampled-noise mask -> organic binary blobs (Perlin-like)."""
    acc = np.zeros((size, size), np.float32)
    for res in (4, 8, 16):
        low = rng.standard_normal((res, res)).astype(np.float32)
        acc += np.asarray(
            Image.fromarray(low).resize((size, size), Image.BICUBIC), np.float32
        )
    acc = (acc - acc.min()) / (np.ptp(acc) + 1e-8)
    thr = rng.uniform(0.6, 0.8)
    return (acc > thr).astype(np.float32)


class DraemDataset(Dataset):
    def __init__(self, paths, size, p_anomaly):
        self.paths, self.size, self.p = paths, size, p_anomaly

    def __len__(self):
        return len(self.paths)

    def _load(self, p):
        im = Image.open(p).convert("L").resize((self.size, self.size), Image.BILINEAR)
        return np.asarray(im, np.float32) / 255.0

    def __getitem__(self, i):
        rng = np.random.default_rng()
        img = self._load(self.paths[i])
        m = np.zeros_like(img)
        aug = img.copy()
        if rng.random() < self.p:
            m = blob_mask(self.size, rng)
            src = self._load(self.paths[rng.integers(len(self.paths))])
            beta = rng.uniform(0.2, 1.0)
            aug = img * (1 - m) + m * (beta * src + (1 - beta) * img)
        t = torch.from_numpy
        return t(img)[None], t(aug.astype(np.float32))[None], t(m)[None]


# --------------------------------------------------------------------------- #
# U-Net
# --------------------------------------------------------------------------- #
def cbr(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class UNet(nn.Module):
    def __init__(self, in_ch, out_ch, c=32):
        super().__init__()
        self.d1, self.d2 = cbr(in_ch, c), cbr(c, c * 2)
        self.d3, self.d4 = cbr(c * 2, c * 4), cbr(c * 4, c * 8)
        self.pool = nn.MaxPool2d(2)
        self.bott = cbr(c * 8, c * 8)
        self.u4, self.u3 = cbr(c * 16, c * 4), cbr(c * 8, c * 2)
        self.u2, self.u1 = cbr(c * 4, c), cbr(c * 2, c)
        self.out = nn.Conv2d(c, out_ch, 1)

    def _up(self, x, skip, blk):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return blk(torch.cat([x, skip], 1))

    def forward(self, x):
        e1 = self.d1(x)
        e2 = self.d2(self.pool(e1))
        e3 = self.d3(self.pool(e2))
        e4 = self.d4(self.pool(e3))
        b = self.bott(self.pool(e4))
        d = self._up(b, e4, self.u4)
        d = self._up(d, e3, self.u3)
        d = self._up(d, e2, self.u2)
        d = self._up(d, e1, self.u1)
        return self.out(d)


# --------------------------------------------------------------------------- #
# losses
# --------------------------------------------------------------------------- #
def gauss_window(ws=11, sigma=1.5):
    g = torch.arange(ws).float() - ws // 2
    g = torch.exp(-(g**2) / (2 * sigma**2))
    g = (g / g.sum())[:, None]
    w = (g @ g.t())[None, None]
    return w


def ssim_loss(x, y, win):
    mux = F.conv2d(x, win, padding=win.shape[-1] // 2)
    muy = F.conv2d(y, win, padding=win.shape[-1] // 2)
    mx2, my2, mxy = mux**2, muy**2, mux * muy
    sx = F.conv2d(x * x, win, padding=win.shape[-1] // 2) - mx2
    sy = F.conv2d(y * y, win, padding=win.shape[-1] // 2) - my2
    sxy = F.conv2d(x * y, win, padding=win.shape[-1] // 2) - mxy
    c1, c2 = 0.01**2, 0.03**2
    ssim = ((2 * mxy + c1) * (2 * sxy + c2)) / ((mx2 + my2 + c1) * (sx + sy + c2))
    return 1 - ssim.mean()


def focal_loss(logits, target, gamma=2.0):
    logp = F.log_softmax(logits, 1)
    t = target.long().squeeze(1)
    logpt = logp.gather(1, t[:, None]).squeeze(1)
    return (-((1 - logpt.exp()) ** gamma) * logpt).mean()


# --------------------------------------------------------------------------- #
def load_img(p, size):
    im = Image.open(p).convert("L").resize((size, size), Image.BILINEAR)
    return torch.from_numpy(np.asarray(im, np.float32) / 255.0)[None]


def load_mask(p, size):
    m = Image.open(p).convert("L").resize((size, size), Image.NEAREST)
    return (np.asarray(m) > 127).astype(np.uint8)


def main() -> None:
    cfg = _cfg
    seed_everything(cfg["seed"])
    if DEVICE == "cuda":
        torch.cuda.set_per_process_memory_fraction(cfg["memory_fraction"], 0)
    raw = Path(cfg["data"]["raw_dir"])
    size = cfg["model"]["input_size"]
    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    def imgs(paths):
        ok = {".jpg", ".png"}
        return [p for p in paths if p.suffix.lower() in ok and p.name != "desktop.ini"]

    train = imgs(sorted(raw.glob(cfg["data"]["train_normal_glob"])))[: cfg["data"]["n_train"]]
    test_norm = imgs(sorted(Path(raw, cfg["data"]["test_normal_dir"]).iterdir()))
    test_abn = imgs(sorted(Path(raw, cfg["data"]["test_abnormal_dir"]).iterdir()))
    mask_dir = Path(raw, cfg["data"]["test_mask_dir"])
    print(f"gpu={cfg['gpu']} train={len(train)} test={len(test_norm)}N/{len(test_abn)}A")

    dl = DataLoader(
        DraemDataset(train, size, cfg["train"]["p_anomaly"]),
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )
    c = cfg["model"]["base_ch"]
    recon = UNet(1, 1, c).to(DEVICE)
    disc = UNet(2, 2, c).to(DEVICE)
    params = list(recon.parameters()) + list(disc.parameters())
    opt = torch.optim.Adam(params, lr=cfg["train"]["lr"])
    win = gauss_window().to(DEVICE)
    sw = cfg["train"]["ssim_weight"]

    for epoch in range(cfg["train"]["epochs"]):
        recon.train()
        disc.train()
        t0, tot = time.time(), 0.0
        for img, aug, m in dl:
            img, aug, m = img.to(DEVICE), aug.to(DEVICE), m.to(DEVICE)
            rec = torch.sigmoid(recon(aug))
            seg = disc(torch.cat([aug, rec], 1))
            loss = F.mse_loss(rec, img) + sw * ssim_loss(rec, img, win) + focal_loss(seg, m)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
        if epoch % 5 == 0 or epoch == cfg["train"]["epochs"] - 1:
            peak = torch.cuda.max_memory_allocated() / 1e9 if DEVICE == "cuda" else 0
            print(f"epoch {epoch:3d} loss {tot / len(dl):.4f} peak_gpu {peak:.1f}GB "
                  f"({time.time() - t0:.0f}s)", flush=True)

    # ---- eval ----
    recon.eval()
    disc.eval()

    @torch.no_grad()
    def anomaly_score(p):
        x = load_img(p, size)[None].to(DEVICE)
        rec = torch.sigmoid(recon(x))
        amap = F.softmax(disc(torch.cat([x, rec], 1)), 1)[:, 1:2]
        smooth = F.avg_pool2d(amap, 21, stride=1, padding=10)
        return smooth[0, 0].cpu(), float(smooth.max())

    ms = cfg["eval"]["map_size"]
    names, y, scores, px_pred, px_true = [], [], [], [], []
    for p in test_norm:
        mp, sc = anomaly_score(p)
        names.append(p.name)
        y.append(0)
        scores.append(sc)
        mp = F.interpolate(mp[None, None], (ms, ms), mode="bilinear", align_corners=False)[0, 0]
        px_pred.append(mp.numpy().reshape(-1))
        px_true.append(np.zeros(ms * ms, np.uint8))
    for p in test_abn:
        mp, sc = anomaly_score(p)
        names.append(p.name)
        y.append(1)
        scores.append(sc)
        mp = F.interpolate(mp[None, None], (ms, ms), mode="bilinear", align_corners=False)[0, 0]
        px_pred.append(mp.numpy().reshape(-1))
        px_true.append(load_mask(mask_dir / f"{p.stem}_mask.jpg", ms).reshape(-1))

    y = np.array(y)
    scores = np.array(scores)
    px_pred, px_true = np.concatenate(px_pred), np.concatenate(px_true)
    metrics = {
        "image_auroc": float(roc_auc_score(y, scores)),
        "image_auprc": float(average_precision_score(y, scores)),
        "pixel_auroc": float(roc_auc_score(px_true, px_pred)),
        "pixel_auprc": float(average_precision_score(px_true, px_pred)),
        "config": cfg,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    with (out_dir / "image_scores.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "label", "score"])
        for nm, yl, sc in zip(names, y, scores, strict=True):
            w.writerow([nm, int(yl), float(sc)])

    print("\n=== DRAEM ===")
    for k in ("image_auroc", "image_auprc", "pixel_auroc", "pixel_auprc"):
        print(f"  {k:12s}: {metrics[k]:.4f}")
    print(f"saved -> {out_dir}")


if __name__ == "__main__":
    main()
