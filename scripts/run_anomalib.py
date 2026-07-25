#!/usr/bin/env python
"""Reproduce literature baselines (FastFlow, CFA, STFPM) via anomalib on MIIC.

Runs each model under our protocol (same fixed test set: 1272 normal + 116
abnormal) and saves per-image scores so the metrics, DeLong test and bootstrap
are directly comparable with our methods. Labels are taken from the folder each
image lives in (robust, independent of anomalib internals).

    uv run --with anomalib python scripts/run_anomalib.py
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "anomalib.yaml"
_cfg = yaml.safe_load(CONFIG_PATH.open())
os.environ["CUDA_VISIBLE_DEVICES"] = str(_cfg["gpu"])

import certifi  # noqa: E402

# uv Python has no system CA bundle -> let backbone-weight downloads verify TLS
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())

import numpy as np  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402


def build_dm(cfg, image_size, bs):
    from anomalib.data import Folder

    d = cfg["data"]
    return Folder(
        name="miic",
        normal_dir=d["train_normal"],
        abnormal_dir=d["abnormal"],
        normal_test_dir=d["normal_test"],
        mask_dir=d["mask"],
        train_batch_size=bs,
        eval_batch_size=bs,
        num_workers=8,
        # keep the full fixed test set: take validation from train, not test
        val_split_mode="from_train",
        val_split_ratio=0.1,
    )


def build_model(name):
    from anomalib.models import Cfa, Fastflow, Stfpm

    return {"stfpm": Stfpm, "fastflow": Fastflow, "cfa": Cfa}[name]()


def epoch_logger(tag):
    """Lightning callback that prints per-epoch progress (rich bar isn't logged)."""
    from lightning.pytorch.callbacks import Callback

    class EpochLog(Callback):
        def on_train_epoch_end(self, trainer, pl_module):
            import time

            print(f"  [{tag}] epoch {trainer.current_epoch + 1}/{trainer.max_epochs}"
                  f"  {time.strftime('%H:%M:%S')}", flush=True)

    return EpochLog()


def extract_scores(preds, abn_names):
    """-> (names, labels, scores) from anomalib predict output."""
    names, labels, scores = [], [], []
    for batch in preds:
        ps = getattr(batch, "pred_score", None)
        ip = getattr(batch, "image_path", None)
        if ps is None or ip is None:  # dict fallback
            ps, ip = batch["pred_score"], batch["image_path"]
        ps = ps.detach().cpu().numpy().reshape(-1)
        ip = ip if isinstance(ip, (list, tuple)) else [ip]
        for s, path in zip(ps, ip, strict=True):
            nm = Path(path).name
            names.append(nm)
            labels.append(1 if nm in abn_names else 0)
            scores.append(float(s))
    return names, np.array(labels), np.array(scores)


def main() -> None:
    from anomalib.engine import Engine

    cfg = _cfg
    import lightning.pytorch as pl

    pl.seed_everything(cfg["seed"])
    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    abn_dir = Path(cfg["data"]["abnormal"])
    abn_names = {p.name for p in abn_dir.iterdir() if p.suffix.lower() == ".jpg"}

    summary = {}
    for name in cfg["models"]:
        print(f"\n===== {name} =====", flush=True)
        dm = build_dm(cfg, cfg["image_size"], cfg["batch_size"])
        model = build_model(name)
        engine = Engine(
            max_epochs=cfg["epochs"][name], accelerator="gpu", devices=1,
            logger=False, callbacks=[epoch_logger(name)],
        )
        engine.fit(model, datamodule=dm)
        preds = engine.predict(model, datamodule=dm)

        names, y, scores = extract_scores(preds, abn_names)
        auroc = float(roc_auc_score(y, scores))
        auprc = float(average_precision_score(y, scores))
        npos = int(y.sum())
        summary[name] = {"image_auroc": auroc, "image_auprc": auprc, "n": len(y), "n_pos": npos}
        print(f"{name}: AUROC {auroc:.4f} AUPRC {auprc:.4f} (n={len(y)}, pos={npos})", flush=True)

        mdir = out_dir / name
        mdir.mkdir(exist_ok=True)
        with (mdir / "image_scores.csv").open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["name", "label", "score"])
            for nm, yl, sc in zip(names, y, scores, strict=True):
                w.writerow([nm, int(yl), float(sc)])

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k:<10} AUROC {v['image_auroc']:.4f}  AUPRC {v['image_auprc']:.4f}")
    print(f"saved -> {out_dir}")


if __name__ == "__main__":
    main()
