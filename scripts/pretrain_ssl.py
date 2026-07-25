#!/usr/bin/env python
"""Domain self-supervised pretraining (DINO) on the MIIC normal images.

Adapts the winning extractor -- the DINOv2 ViT-S -- to the IC-microscopy domain
with the DINO objective (no labels). The delicate DINO pieces (centering loss,
weight-normalised head, schedules) come from the vetted ``lightly`` library; the
training loop is ours so we control batching and GPU memory explicitly.

Resource policy (the priority here is NOT to saturate the shared server):
  * pins ONE GPU (``gpu`` in the config) and hard-caps its memory (``memory_fraction``);
  * gradient checkpointing on the student ViT + mixed precision to keep VRAM low;
  * a modest batch size; peak GPU memory is printed every epoch.

Reproducible & argument-free: reads ``configs/ssl.yaml``.

    uv run python scripts/pretrain_ssl.py
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path

import certifi
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "ssl.yaml"

# read config and pin the GPU BEFORE importing torch
_cfg = yaml.safe_load(CONFIG_PATH.open())
os.environ["CUDA_VISIBLE_DEVICES"] = str(_cfg["gpu"])
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())

import timm  # noqa: E402
import torch  # noqa: E402
from lightly.loss import DINOLoss  # noqa: E402
from lightly.models.modules import DINOProjectionHead  # noqa: E402
from lightly.models.utils import deactivate_requires_grad, update_momentum  # noqa: E402
from lightly.transforms.dino_transform import DINOTransform  # noqa: E402
from PIL import Image  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from semimil.utils import seed_everything  # noqa: E402


class MultiCropNormals(Dataset):
    def __init__(self, paths: list[Path], transform):
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        return self.transform(Image.open(self.paths[i]).convert("RGB"))  # list of crops


def collate_multicrop(batch):
    """batch: list of B items, each a list of N crops -> list of N tensors [B,3,Hi,Wi]."""
    n = len(batch[0])
    return [torch.stack([item[i] for item in batch]) for i in range(n)]


class DINOModel(nn.Module):
    """DINOv2 ViT backbone (CLS token) + DINO projection head."""

    def __init__(self, backbone: nn.Module, head: DINOProjectionHead):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        cls = self.backbone.forward_features(x)[:, 0]  # CLS token
        return self.head(cls)


def cosine(step, total, base, final):
    return final + 0.5 * (base - final) * (1 + math.cos(math.pi * step / max(1, total)))


def main() -> None:
    cfg = _cfg
    seed_everything(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.set_per_process_memory_fraction(cfg["memory_fraction"], 0)
    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = Path(cfg["data"]["raw_dir"])
    paths = [
        p for p in sorted(raw.glob(cfg["data"]["train_normal_glob"])) if p.name != "desktop.ini"
    ]
    if cfg["data"]["limit_images"]:
        paths = paths[: cfg["data"]["limit_images"]]
    print(f"gpu={cfg['gpu']}  mem_cap={cfg['memory_fraction']}  images={len(paths)}", flush=True)

    cc = cfg["crops"]
    transform = DINOTransform(
        global_crop_size=cc["global_size"],
        local_crop_size=cc["local_size"],
        n_local_views=cc["n_local"],
    )
    ds = MultiCropNormals(paths, transform)
    dl = DataLoader(
        ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        collate_fn=collate_multicrop,
    )

    # --- student / teacher (built separately; deepcopy breaks on weight_norm) ---
    mc = cfg["model"]

    def build(grad_ckpt: bool) -> DINOModel:
        bb = timm.create_model(
            mc["backbone"], pretrained=True, num_classes=0, dynamic_img_size=True
        )
        bb.set_grad_checkpointing(grad_ckpt)
        head = DINOProjectionHead(
            bb.embed_dim, mc["proj_hidden"], mc["bottleneck"], mc["out_dim"],
            freeze_last_layer=cfg["train"]["freeze_last_layer_epochs"],
        )
        return DINOModel(bb, head).to(device)

    student = build(cfg["train"]["grad_checkpointing"])
    teacher = build(False)
    teacher.load_state_dict(student.state_dict())  # identical init
    deactivate_requires_grad(teacher)

    loss_fn = DINOLoss(
        output_dim=mc["out_dim"],
        warmup_teacher_temp=cfg["train"]["warmup_teacher_temp"],
        teacher_temp=cfg["train"]["teacher_temp"],
        warmup_teacher_temp_epochs=cfg["train"]["warmup_teacher_temp_epochs"],
    ).to(device)

    opt = torch.optim.AdamW(
        student.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"]
    )

    steps_per_epoch = len(dl)
    total = cfg["train"]["epochs"] * steps_per_epoch
    warmup = cfg["train"]["warmup_epochs"] * steps_per_epoch
    base_lr, base_m = cfg["train"]["lr"], cfg["train"]["base_momentum"]
    print(f"steps/epoch={steps_per_epoch}  total={total}  crops=2+{cc['n_local']}", flush=True)

    gstep = 0
    for epoch in range(cfg["train"]["epochs"]):
        student.train()
        torch.cuda.reset_peak_memory_stats()
        t0, running = time.time(), 0.0
        for views in dl:
            views = [v.to(device, non_blocking=True) for v in views]
            globals_ = views[:2]
            if gstep < warmup:
                lr = base_lr * (gstep / max(1, warmup))
            else:
                lr = cosine(gstep - warmup, total - warmup, base_lr, 1e-6)
            for g in opt.param_groups:
                g["lr"] = lr
            # bf16 (A100): stable for DINO's large-output softmax, no GradScaler needed
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                with torch.no_grad():
                    teacher_out = [teacher(g) for g in globals_]
                student_out = [student(v) for v in views]
                # NB: pass epoch as keyword -- the 3rd positional arg is teacher_temp
                loss = loss_fn(teacher_out, student_out, epoch=epoch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 3.0)
            student.head.cancel_last_layer_gradients(epoch)
            opt.step()
            m = cosine(gstep, total, base_m, 1.0)
            update_momentum(student, teacher, m)
            running += loss.item()
            gstep += 1
        peak = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0
        ips = steps_per_epoch * cfg["train"]["batch_size"] / (time.time() - t0)
        print(
            f"epoch {epoch:3d}  loss {running / steps_per_epoch:.4f}  lr {lr:.2e}  "
            f"peak_gpu {peak:.1f}GB  {ips:.0f} img/s",
            flush=True,
        )
        if (epoch + 1) % cfg["output"]["ckpt_every"] == 0 or epoch == cfg["train"]["epochs"] - 1:
            ckpt = out_dir / "dinov2_domain.pth"
            torch.save(student.backbone.state_dict(), ckpt)
            print(f"  saved {ckpt}", flush=True)

    print("done.", flush=True)


if __name__ == "__main__":
    main()
