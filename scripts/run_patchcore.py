#!/usr/bin/env python
"""One-class detector -- MIL over a memory bank (PatchCore-style) on MIIC.

Framed as MIL: each 512x512 image is a *bag*; each feature-map cell is an
*instance*. We model "normal" with a memory bank of normal patch features, score
every instance by its distance to that bank, and pool instance scores into a bag
score (top-k mean) -- the MIL ">=1 positive instance" assumption. No labels and
no training are used; normality is the model.

Reproducible & argument-free: all settings come from ``configs/patchcore.yaml``.

    uv run python scripts/run_patchcore.py
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import certifi
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from scipy.ndimage import gaussian_filter
from sklearn.metrics import average_precision_score, roc_auc_score

from semimil.utils import seed_everything

# The uv-managed Python has no system CA bundle -> point urllib/torch.hub at
# certifi so backbone-weight downloads (WRN50, DINOv2, ...) verify TLS.
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "patchcore.yaml"

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ----------------------------------------------------------------------------- #
# data
# ----------------------------------------------------------------------------- #
def list_images(paths: list[Path]) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    return [p for p in paths if p.suffix.lower() in exts and p.name != "desktop.ini"]


def load_image(path: Path, size: int) -> torch.Tensor:
    """Grayscale PNG/JPG -> normalized 3-channel tensor [3, H, W]."""
    im = Image.open(path).convert("L").resize((size, size), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0)  # [H,W]
    x = x.unsqueeze(0).repeat(3, 1, 1)  # gray -> 3ch
    return (x - IMAGENET_MEAN[0]) / IMAGENET_STD[0]


def load_mask(path: Path, size: int) -> np.ndarray:
    m = Image.open(path).convert("L").resize((size, size), Image.NEAREST)
    return (np.asarray(m) > 127).astype(np.uint8)


# ----------------------------------------------------------------------------- #
# feature extractor
# ----------------------------------------------------------------------------- #
class PatchFeatureExtractor:
    """Multi-scale patch embeddings from an ImageNet backbone (PatchCore-style)."""

    def __init__(self, backbone: str, layers: list[str], device: str):
        import torchvision.models as tvm

        weights = {"wide_resnet50_2": tvm.Wide_ResNet50_2_Weights.IMAGENET1K_V1}[backbone]
        self.model = getattr(tvm, backbone)(weights=weights).eval().to(device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = device
        self._feats: dict[str, torch.Tensor] = {}
        for name in layers:
            getattr(self.model, name).register_forward_hook(self._hook(name))
        self.layers = layers

    def _hook(self, name: str):
        def fn(_m, _i, out):
            self._feats[name] = out

        return fn

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """[B,3,H,W] -> [B, Hf, Wf, D] concatenated multi-scale features."""
        self._feats.clear()
        self.model(x.to(self.device))
        ref = self._feats[self.layers[0]]
        hf, wf = ref.shape[-2:]
        pooled = []
        for name in self.layers:
            f = self._feats[name]
            f = F.avg_pool2d(f, 3, stride=1, padding=1)  # local neighborhood agg
            f = F.interpolate(f, size=(hf, wf), mode="bilinear", align_corners=False)
            pooled.append(f)
        f = torch.cat(pooled, dim=1)  # [B, D, Hf, Wf]
        return f.permute(0, 2, 3, 1).contiguous()  # [B, Hf, Wf, D]


class DINOv2FeatureExtractor:
    """Patch-token embeddings from a DINOv2 ViT (frozen, no training).

    If ``weights`` is given, load a domain-adapted checkpoint (from
    ``pretrain_ssl.py``) over the off-the-shelf DINOv2 backbone.
    """

    def __init__(self, name: str, device: str, weights: str | None = None):
        import timm

        # dynamic_img_size lets the ViT accept any multiple-of-14 input (pos-embed
        # interpolation) -- needed for the resolution ablation.
        self.model = timm.create_model(
            name, pretrained=True, num_classes=0, dynamic_img_size=True
        ).to(device)
        if weights:
            state = torch.load(weights, map_location=device)
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            print(f"[dinov2] loaded domain weights {weights}")
            print(f"[dinov2]   missing={len(missing)} unexpected={len(unexpected)}")
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = device
        self.prefix = self.model.num_prefix_tokens  # cls + register tokens
        self.patch = self.model.patch_embed.patch_size[0]

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """[B,3,H,W] -> [B, Hf, Wf, D] patch tokens on their spatial grid."""
        x = x.to(self.device)
        g = x.shape[-1] // self.patch
        tokens = self.model.forward_features(x)  # [B, prefix + g*g, D]
        patch = tokens[:, self.prefix :, :]  # drop cls/register tokens
        return patch.reshape(patch.shape[0], g, g, patch.shape[-1]).contiguous()


def build_extractor(cfg_model: dict, device: str):
    """Dispatch on ``model.type`` -> CNN (PatchCore) or DINOv2 feature extractor."""
    kind = cfg_model.get("type", "cnn")
    if kind == "cnn":
        return PatchFeatureExtractor(cfg_model["backbone"], cfg_model["layers"], device)
    if kind == "dinov2":
        return DINOv2FeatureExtractor(cfg_model["dino_name"], device, cfg_model.get("weights"))
    raise ValueError(f"unknown model.type: {kind!r}")


def batched(items: list, n: int):
    for i in range(0, len(items), n):
        yield items[i : i + n]


# ----------------------------------------------------------------------------- #
# scoring
# ----------------------------------------------------------------------------- #
@torch.no_grad()
def min_dist_to_memory(feats: torch.Tensor, memory: torch.Tensor, chunk: int = 512) -> torch.Tensor:
    """Nearest-neighbour distance of each instance to the memory bank -> [N]."""
    out = torch.empty(feats.shape[0], device=feats.device)
    for i in range(0, feats.shape[0], chunk):
        d = torch.cdist(feats[i : i + chunk], memory)  # [chunk, K]
        out[i : i + chunk] = d.min(dim=1).values
    return out


def main() -> None:
    cfg = yaml.safe_load(CONFIG_PATH.open())
    seed_everything(cfg["seed"])
    rng = random.Random(cfg["seed"])
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

    # --- build memory bank from normal images ---------------------------------
    train_normals = list_images(sorted(raw.glob(cfg["data"]["train_normal_glob"])))
    rng.shuffle(train_normals)
    mem_imgs = train_normals[: cfg["memory"]["n_images"]]
    print(f"[memory] {len(mem_imgs)} normal images -> bank")
    ppi = cfg["memory"]["patches_per_image"]
    bank = []
    t0 = time.time()
    for batch in batched(mem_imgs, bs):
        x = torch.stack([load_image(p, size) for p in batch])
        feats = extractor(x)  # [B, Hf, Wf, D]
        feats = feats.reshape(feats.shape[0], -1, feats.shape[-1])  # [B, Hf*Wf, D]
        for b in range(feats.shape[0]):
            idx = torch.randint(0, feats.shape[1], (ppi,))
            bank.append(feats[b, idx].cpu())
    bank_t = torch.cat(bank, dim=0)
    if bank_t.shape[0] > cfg["memory"]["coreset_size"]:
        sel = torch.randperm(bank_t.shape[0])[: cfg["memory"]["coreset_size"]]
        bank_t = bank_t[sel]
    memory = bank_t.to(device).float()
    print(f"[memory] bank shape={tuple(memory.shape)}  ({time.time() - t0:.1f}s)")

    # --- score test set -------------------------------------------------------
    test_normals = list_images(sorted(Path(raw, cfg["data"]["test_normal_dir"]).iterdir()))
    test_abnormals = list_images(sorted(Path(raw, cfg["data"]["test_abnormal_dir"]).iterdir()))
    mask_dir = Path(raw, cfg["data"]["test_mask_dir"])
    map_size = cfg["eval"]["map_size"]
    top_k = cfg["score"]["top_k"]

    def score_set(paths, is_abnormal):
        img_scores, maps, masks, names = [], [], [], []
        for batch in batched(paths, bs):
            x = torch.stack([load_image(p, size) for p in batch])
            feats = extractor(x)  # [B,Hf,Wf,D]
            hf, wf, d = feats.shape[1:]
            for b, p in enumerate(batch):
                dist = min_dist_to_memory(feats[b].reshape(-1, d), memory)  # [Hf*Wf]
                amap = dist.reshape(hf, wf)
                img_scores.append(torch.topk(dist, top_k).values.mean().item())
                small = (
                    F.interpolate(
                        amap[None, None],
                        size=(map_size, map_size),
                        mode="bilinear",
                        align_corners=False,
                    )[0, 0]
                    .cpu()
                    .numpy()
                )
                maps.append(small)
                names.append(p.name)
                if is_abnormal:
                    mpath = mask_dir / f"{p.stem}_mask.jpg"
                    masks.append(load_mask(mpath, map_size))
                else:
                    masks.append(np.zeros((map_size, map_size), np.uint8))
        return img_scores, maps, masks, names

    t0 = time.time()
    ns, nmaps, nmasks, nnames = score_set(test_normals, False)
    as_, amaps, amasks, anames = score_set(test_abnormals, True)
    print(f"[score] {len(ns)} normal + {len(as_)} abnormal  ({time.time() - t0:.1f}s)")

    # --- metrics --------------------------------------------------------------
    y_img = np.array([0] * len(ns) + [1] * len(as_))
    s_img = np.array(ns + as_)
    px_pred = np.stack(nmaps + amaps).reshape(-1)
    px_true = np.stack(nmasks + amasks).reshape(-1)
    metrics = {
        "run_name": cfg["run_name"],
        "model_type": cfg["model"].get("type", "cnn"),
        "image_auroc": float(roc_auc_score(y_img, s_img)),
        "image_auprc": float(average_precision_score(y_img, s_img)),
        "pixel_auroc": float(roc_auc_score(px_true, px_pred)),
        "pixel_auprc": float(average_precision_score(px_true, px_pred)),
        "n_test_normal": len(ns),
        "n_test_abnormal": len(as_),
        "memory_bank": list(memory.shape),
        "config": cfg,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    import csv

    with (out_dir / "image_scores.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "label", "score"])
        for n, s in zip(nnames, ns, strict=True):
            w.writerow([n, 0, s])
        for n, s in zip(anames, as_, strict=True):
            w.writerow([n, 1, s])

    # --- localization figures -------------------------------------------------
    save_viz(out_dir, test_abnormals, amaps, amasks, cfg, size)

    print("\n=== RESULTS ===")
    for k in ("image_auroc", "image_auprc", "pixel_auroc", "pixel_auprc"):
        print(f"  {k:14s}: {metrics[k]:.4f}")
    print(f"\nsaved -> {out_dir}")


def save_viz(out_dir, abn_paths, amaps, amasks, cfg, size) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = min(cfg["eval"]["n_viz"], len(abn_paths))
    sigma = cfg["score"]["gaussian_sigma"]
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = axes[None]
    for i in range(n):
        img = np.asarray(Image.open(abn_paths[i]).convert("L").resize((size, size)))
        heat = gaussian_filter(
            np.asarray(Image.fromarray(amaps[i]).resize((size, size), Image.BILINEAR)), sigma
        )
        mask = np.asarray(Image.fromarray(amasks[i] * 255).resize((size, size), Image.NEAREST))
        axes[i, 0].imshow(img, cmap="gray")
        axes[i, 0].set_title("image")
        axes[i, 1].imshow(img, cmap="gray")
        axes[i, 1].imshow(heat, cmap="jet", alpha=0.5)
        axes[i, 1].set_title("anomaly map")
        axes[i, 2].imshow(mask, cmap="gray")
        axes[i, 2].set_title("GT mask")
        for a in axes[i]:
            a.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "localization.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
