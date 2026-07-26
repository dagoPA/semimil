#!/usr/bin/env python
"""Reproduce PatchCore (anomalib) on MIIC under our clean, leakage-free protocol.

Adds the canonical memory-bank baseline to the unified comparison. Like our
one-class detector, PatchCore does not train over epochs, so it should reproduce
its reported number (~0.933) under a leakage-free protocol -- confirming that our
harness is faithful and that the gap of the trainable baselines comes from
checkpoint selection on the test set, not from the reproduction.

    uv run --with anomalib --with certifi python scripts/run_patchcore_anomalib.py
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
_cfg = yaml.safe_load((REPO_ROOT / "configs" / "anomalib.yaml").open())
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import certifi  # noqa: E402

os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())

import numpy as np  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402


def main() -> None:
    import lightning.pytorch as pl
    from anomalib.data import Folder
    from anomalib.engine import Engine
    from anomalib.models import Patchcore

    cfg = _cfg
    pl.seed_everything(cfg["seed"])
    d = cfg["data"]
    # PatchCore subsamples a memory bank anyway; a 1000-image subset + a 5% coreset
    # keeps GPU memory well within a shared A100 while still reproducing its number.
    train_dir = str(Path(d["train_normal"]).parent / "train_normal_1k")
    dm = Folder(
        name="miic", normal_dir=train_dir, abnormal_dir=d["abnormal"],
        normal_test_dir=d["normal_test"], mask_dir=d["mask"],
        train_batch_size=cfg["batch_size"], eval_batch_size=cfg["batch_size"],
        num_workers=8, val_split_mode="from_train", val_split_ratio=0.1,
    )
    model = Patchcore(coreset_sampling_ratio=0.05)  # WRN50 backbone
    out = Path(cfg["output"]["dir"]) / "patchcore"
    out.mkdir(parents=True, exist_ok=True)
    engine = Engine(max_epochs=1, accelerator="gpu", devices=1, logger=False,
                    default_root_dir=str(out / "_ckpt"))
    engine.fit(model, datamodule=dm)
    preds = engine.predict(model, datamodule=dm)

    abn = {p.name for p in Path(d["abnormal"]).iterdir() if p.suffix.lower() == ".jpg"}
    names, labels, scores = [], [], []
    for batch in preds:
        ps = getattr(batch, "pred_score", None)
        ip = getattr(batch, "image_path", None)
        if ps is None:
            ps, ip = batch["pred_score"], batch["image_path"]
        ps = ps.detach().cpu().numpy().reshape(-1)
        ip = ip if isinstance(ip, (list, tuple)) else [ip]
        for s, path in zip(ps, ip, strict=True):
            nm = Path(path).name
            names.append(nm)
            labels.append(1 if nm in abn else 0)
            scores.append(float(s))
    y = np.array(labels)
    s = np.array(scores)
    print(f"PatchCore: AUROC {roc_auc_score(y, s):.4f} AUPRC "
          f"{average_precision_score(y, s):.4f} (n={len(y)}, pos={int(y.sum())})", flush=True)
    with (out / "image_scores.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "label", "score"])
        for nm, yl, sc in zip(names, y, scores, strict=True):
            w.writerow([nm, int(yl), float(sc)])
    print(f"saved -> {out / 'image_scores.csv'}", flush=True)


if __name__ == "__main__":
    main()
