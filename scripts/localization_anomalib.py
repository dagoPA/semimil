#!/usr/bin/env python
"""Pixel AUROC + PRO for the anomalib baselines (STFPM, FastFlow, CFA).

Re-fits each model under our clean protocol (val from train), predicts on the
real test set, extracts the per-image anomaly map, and scores localization with
the SAME pixel-AUROC + PRO functions used for Track A (scripts/localization.py),
so the localization table is computed uniformly across methods.

    uv run --with anomalib --with certifi python scripts/localization_anomalib.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
_cfg = yaml.safe_load((REPO_ROOT / "configs" / "anomalib.yaml").open())
os.environ["CUDA_VISIBLE_DEVICES"] = str(_cfg["gpu"])

import certifi  # noqa: E402

os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())

import numpy as np  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from localization import pro_auc  # noqa: E402

MAP = 256
ROOT = Path("/autofs/space/crater_001/projects/micropath/semi")
OUT = ROOT / "outputs/localization"
LABEL = {"stfpm": "STFPM", "fastflow": "FastFlow", "cfa": "CFA"}


def build_dm(cfg):
    from anomalib.data import Folder

    d = cfg["data"]
    return Folder(
        name="miic", normal_dir=d["train_normal"], abnormal_dir=d["abnormal"],
        normal_test_dir=d["normal_test"], mask_dir=d["mask"],
        train_batch_size=cfg["batch_size"], eval_batch_size=cfg["batch_size"],
        num_workers=8, val_split_mode="from_train", val_split_ratio=0.1,
    )


def build_model(name):
    from anomalib.models import Cfa, Fastflow, Stfpm

    return {"stfpm": Stfpm, "fastflow": Fastflow, "cfa": Cfa}[name]()


def load_mask(path: Path) -> np.ndarray:
    m = Image.open(path).convert("L").resize((MAP, MAP), Image.NEAREST)
    return (np.asarray(m) > 127).astype(np.uint8)


def to_map(am) -> np.ndarray:
    """anomalib anomaly_map tensor -> [MAP,MAP] float numpy."""
    t = am.detach().float().cpu()
    while t.dim() > 2:
        t = t.squeeze(0)
    up = F.interpolate(t[None, None], size=(MAP, MAP), mode="bilinear",
                       align_corners=False)[0, 0]
    return up.numpy()


def main() -> None:
    import lightning.pytorch as pl
    from anomalib.engine import Engine

    cfg = _cfg
    pl.seed_everything(cfg["seed"])
    OUT.mkdir(parents=True, exist_ok=True)
    abn_names = {p.name for p in Path(cfg["data"]["abnormal"]).iterdir()
                 if p.suffix.lower() == ".jpg"}
    mask_dir = Path(cfg["data"]["mask"])

    results = []
    for name in cfg["models"]:
        print(f"\n===== {name} =====", flush=True)
        dm = build_dm(cfg)
        model = build_model(name)
        # keep anomalib's checkpoints on the big shared FS (home has a quota)
        engine = Engine(max_epochs=cfg["epochs"][name], accelerator="gpu",
                        devices=1, logger=False,
                        default_root_dir=str(OUT / "_ckpt"))
        engine.fit(model, datamodule=dm)
        preds = engine.predict(model, datamodule=dm)

        maps, masks = [], []
        for batch in preds:
            am = getattr(batch, "anomaly_map", None)
            ip = getattr(batch, "image_path", None)
            if am is None:
                am, ip = batch["anomaly_map"], batch["image_path"]
            ip = ip if isinstance(ip, (list, tuple)) else [ip]
            am = am if am.dim() > 2 else am[None]
            for j, path in enumerate(ip):
                maps.append(to_map(am[j]))
                nm = Path(path).name
                masks.append(load_mask(mask_dir / f"{Path(nm).stem}_mask.jpg")
                             if nm in abn_names else np.zeros((MAP, MAP), np.uint8))
        M = np.stack(maps)
        G = np.stack(masks)
        px_auroc = float(roc_auc_score(G.ravel(), M.ravel()))
        px_auprc = float(average_precision_score(G.ravel(), M.ravel()))
        pro = pro_auc(M, G)
        results.append({"method": LABEL[name], "run": name, "pixel_auroc": px_auroc,
                        "pixel_auprc": px_auprc, "pro": pro, "map_res": MAP,
                        "n": int(G.shape[0])})
        print(f"{name}: pixel AUROC {px_auroc:.4f} AUPRC {px_auprc:.4f} | PRO {pro:.4f} "
              f"(n={G.shape[0]})", flush=True)
        (OUT / "anomalib.json").write_text(json.dumps(results, indent=2))  # incremental

    (OUT / "anomalib.json").write_text(json.dumps(results, indent=2))
    print(f"\nsaved -> {OUT / 'anomalib.json'}")


if __name__ == "__main__":
    main()
