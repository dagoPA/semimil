#!/usr/bin/env python
"""Isolate 'best-checkpoint-on-test' as the source of the reproduction gap.

Literature [14] reports high image AUROC on MIIC (CFA 0.99, FastFlow 0.976,
STFPM 0.963) using anomalib's DEFAULT split, where HALF the test set -- real
labeled anomalies included -- is used as a validation set to save the best model
during training. Our main runs deliberately avoid this (val from train, last
epoch), and score far lower.

This script holds our clean setup fixed and adds ONLY per-epoch model selection:
it validates on the FULL test set every epoch (val_split_mode=same_as_test) and
records the image AUROC curve. We then report LAST-epoch (leakage-free, = our
main number) vs BEST-epoch (their-style optimistic selection). If BEST >> LAST
and approaches the literature values, epoch selection on test is the cause.

    uv run --with anomalib --with certifi python scripts/anomalib_epochsel.py
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "anomalib_epochsel.yaml"
_cfg = yaml.safe_load(CONFIG_PATH.open())
os.environ["CUDA_VISIBLE_DEVICES"] = str(_cfg["gpu"])

import certifi  # noqa: E402

# uv Python has no system CA bundle -> let backbone-weight downloads verify TLS
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())

import numpy as np  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

# literature-reported image AUROC on MIIC (Barusco et al., 2025), for reference
LIT = {"stfpm": 0.9632, "fastflow": 0.9763, "cfa": 0.9901}


def build_dm(cfg, bs):
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
        # validate on the FULL test set every epoch so we get a per-epoch AUROC
        # curve (test set stays full for the final predict too).
        val_split_mode="same_as_test",
    )


def build_model(name):
    from anomalib.models import Cfa, Fastflow, Stfpm

    return {"stfpm": Stfpm, "fastflow": Fastflow, "cfa": Cfa}[name]()


def _grab(obj, *attrs):
    """Fetch the first present attribute/key from an anomalib Batch or dict."""
    for a in attrs:
        v = getattr(obj, a, None)
        if v is None and isinstance(obj, dict):
            v = obj.get(a)
        if v is not None:
            return v
    return None


def curve_logger(tag, curve):
    """Compute image AUROC on the (full-test) validation set each epoch.

    anomalib only logs AUROC at test time, so we accumulate per-batch
    pred_score + gt_label ourselves and score with sklearn -- robust to the
    library's internal metric naming.
    """
    import time

    from lightning.pytorch.callbacks import Callback
    from sklearn.metrics import roc_auc_score

    class Rec(Callback):
        def __init__(self):
            self.s: list = []
            self.y: list = []

        def on_validation_epoch_start(self, trainer, pl_module):
            self.s, self.y = [], []

        def on_validation_batch_end(self, trainer, pl_module, outputs, batch,
                                    batch_idx, dataloader_idx=0):
            ps = _grab(outputs, "pred_score")
            gt = _grab(batch, "gt_label", "gt_labels")
            if gt is None:
                gt = _grab(outputs, "gt_label", "gt_labels")
            if ps is None or gt is None:
                return
            self.s.append(np.asarray(ps.detach().cpu()).reshape(-1))
            self.y.append(np.asarray(gt.detach().cpu()).reshape(-1))

        def on_validation_epoch_end(self, trainer, pl_module):
            if trainer.sanity_checking or not self.s:
                return
            s = np.concatenate(self.s)
            y = np.concatenate(self.y).astype(int)
            au = float(roc_auc_score(y, s)) if y.min() != y.max() else None
            curve.append({"epoch": int(trainer.current_epoch), "auroc": au,
                          "n": int(y.size), "n_pos": int(y.sum())})
            print(f"  [{tag}] epoch {trainer.current_epoch + 1}  "
                  f"img_AUROC={au if au is None else round(au, 4)}  "
                  f"(n={y.size}, pos={int(y.sum())})  {time.strftime('%H:%M:%S')}",
                  flush=True)

    return Rec()


def _pick_auroc(row):
    return row.get("auroc")


def extract_scores(preds, abn_names):
    names, labels, scores = [], [], []
    for batch in preds:
        ps = getattr(batch, "pred_score", None)
        ip = getattr(batch, "image_path", None)
        if ps is None or ip is None:
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
    import lightning.pytorch as pl
    from anomalib.engine import Engine

    cfg = _cfg
    pl.seed_everything(cfg["seed"])
    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    abn_names = {p.name for p in Path(cfg["data"]["abnormal"]).iterdir()
                 if p.suffix.lower() == ".jpg"}

    summary = {}
    for name in cfg["models"]:
        print(f"\n===== {name} (same_as_test, per-epoch AUROC) =====", flush=True)
        curve: list[dict] = []
        dm = build_dm(cfg, cfg["batch_size"])
        model = build_model(name)
        engine = Engine(
            max_epochs=cfg["epochs"][name], accelerator="gpu", devices=1,
            logger=False,
            callbacks=[curve_logger(name, curve)],
        )
        engine.fit(model, datamodule=dm)

        # last-epoch scoring (leakage-free number, matches configs/anomalib.yaml)
        preds = engine.predict(model, datamodule=dm)
        names, y, scores = extract_scores(preds, abn_names)
        last_auroc = float(roc_auc_score(y, scores))
        last_auprc = float(average_precision_score(y, scores))

        aurocs = [(_pick_auroc(r), r["epoch"]) for r in curve]
        aurocs = [(a, e) for a, e in aurocs if a is not None]
        best_auroc, best_epoch = max(aurocs, key=lambda t: t[0]) if aurocs else (None, None)

        summary[name] = {
            "last_epoch_auroc": last_auroc,
            "last_epoch_auprc": last_auprc,
            "best_epoch_auroc": best_auroc,
            "best_epoch": best_epoch,
            "literature_auroc": LIT.get(name),
            "n": len(y), "n_pos": int(y.sum()),
        }
        print(f"{name}: LAST {last_auroc:.4f} | BEST {best_auroc:.4f} "
              f"(ep {best_epoch}) | LIT {LIT.get(name)}", flush=True)

        mdir = out_dir / name
        mdir.mkdir(exist_ok=True)
        (mdir / "curve.json").write_text(json.dumps(curve, indent=2))
        with (mdir / "image_scores_last.csv").open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["name", "label", "score"])
            for nm, yl, sc in zip(names, y, scores, strict=True):
                w.writerow([nm, int(yl), float(sc)])

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== EPOCH-SELECTION SUMMARY (image AUROC) ===")
    print(f"  {'model':<10}{'last':>8}{'best':>8}{'lit':>8}")
    for k, v in summary.items():
        lit = v["literature_auroc"]
        print(f"  {k:<10}{v['last_epoch_auroc']:>8.3f}{v['best_epoch_auroc']:>8.3f}"
              f"{lit:>8.3f}")
    print(f"saved -> {out_dir}")


if __name__ == "__main__":
    main()
