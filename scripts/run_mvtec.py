#!/usr/bin/env python
"""One-class detector (DINOv2, no training) on MVTec-AD -- external validity.

Runs the exact MIIC one-class method on each of the 15 MVTec categories: build a
memory bank from that category's train/good images, score its test split by
nearest-neighbour distance, and report image AUROC + pixel AUROC + PRO. The mean
over categories tests whether the method generalizes beyond the IC-microscopy
domain. No labels and no training are used.

    uv run python scripts/run_mvtec.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
_cfg = yaml.safe_load((REPO_ROOT / "configs" / "mvtec.yaml").open())
os.environ["CUDA_VISIBLE_DEVICES"] = str(_cfg["gpu"])

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from localization import pro_auc  # noqa: E402
from run_patchcore import DINOv2FeatureExtractor, batched, min_dist_to_memory  # noqa: E402

from semimil.utils import seed_everything  # noqa: E402

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def load_rgb(path: Path, size: int) -> torch.Tensor:
    im = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(im, np.float32) / 255.0).permute(2, 0, 1)
    return (x - MEAN) / STD


def load_mask(path: Path, size: int) -> np.ndarray:
    if not path.exists():
        return np.zeros((size, size), np.uint8)
    m = Image.open(path).convert("L").resize((size, size), Image.NEAREST)
    return (np.asarray(m) > 127).astype(np.uint8)


def main() -> None:
    cfg = _cfg
    seed_everything(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path(cfg["root"])
    size = cfg["input_size"]
    bs = cfg["batch_size"]
    mres = cfg["map_res"]
    topk = cfg["score"]["top_k"]
    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    cats = sorted(p.name for p in root.iterdir()
                  if (p / "train" / "good").is_dir())
    print(f"device={device}  {len(cats)} categories: {cats}", flush=True)

    extractor = DINOv2FeatureExtractor(cfg["model"]["dino_name"], device)

    def embed(paths):
        feats = []
        for batch in batched(paths, bs):
            x = torch.stack([load_rgb(p, size) for p in batch])
            f = extractor(x)  # [B,Hf,Wf,D]
            feats.append(f)
        return feats

    results = {}
    rng = np.random.default_rng(cfg["seed"])
    for cat in cats:
        t0 = time.time()
        # --- memory bank from train/good ---
        good = sorted((root / cat / "train" / "good").glob("*.png"))
        rng.shuffle(good)
        good = good[: cfg["memory"]["max_images"]]
        ppi = cfg["memory"]["patches_per_image"]
        bank = []
        for f in embed(good):
            f = f.reshape(f.shape[0], -1, f.shape[-1])
            for b in range(f.shape[0]):
                idx = torch.randint(0, f.shape[1], (ppi,))
                bank.append(f[b, idx].cpu())
        bank_t = torch.cat(bank, 0)
        if bank_t.shape[0] > cfg["memory"]["coreset_size"]:
            sel = torch.randperm(bank_t.shape[0])[: cfg["memory"]["coreset_size"]]
            bank_t = bank_t[sel]
        memory = bank_t.to(device).float()

        # --- score test split ---
        img_scores, labels, maps, masks = [], [], [], []
        for defect in sorted((root / cat / "test").iterdir()):
            gt_dir = root / cat / "ground_truth" / defect.name
            is_good = defect.name == "good"
            for p in sorted(defect.glob("*.png")):
                x = load_rgb(p, size)[None]
                feats = extractor(x)
                hf, wf, d = feats.shape[1:]
                dist = min_dist_to_memory(feats[0].reshape(-1, d), memory)
                img_scores.append(torch.topk(dist, topk).values.mean().item())
                labels.append(0 if is_good else 1)
                up = F.interpolate(dist.reshape(hf, wf)[None, None], size=(mres, mres),
                                   mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
                maps.append(up)
                masks.append(np.zeros((mres, mres), np.uint8) if is_good
                             else load_mask(gt_dir / f"{p.stem}_mask.png", mres))

        y = np.array(labels)
        s = np.array(img_scores)
        M = np.stack(maps)
        G = np.stack(masks)
        img_auroc = float(roc_auc_score(y, s))
        px_auroc = float(roc_auc_score(G.ravel(), M.ravel()))
        pro = pro_auc(M, G)
        results[cat] = {"image_auroc": img_auroc, "pixel_auroc": px_auroc, "pro": pro,
                        "n_test": int(len(y)), "n_pos": int(y.sum())}
        print(f"  {cat:<12} img {img_auroc:.4f} | pixel {px_auroc:.4f} | PRO {pro:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)

    mean = {k: float(np.mean([results[c][k] for c in cats]))
            for k in ("image_auroc", "pixel_auroc", "pro")}
    summary = {"mean": mean, "per_category": results, "n_categories": len(cats)}
    (out_dir / "results.json").write_text(json.dumps(summary, indent=2))
    print("\n=== MVTec-AD mean (one-class detector, DINOv2, no training) ===")
    print(f"  image AUROC {mean['image_auroc']:.4f} | pixel AUROC {mean['pixel_auroc']:.4f} "
          f"| PRO {mean['pro']:.4f}")
    print(f"saved -> {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
