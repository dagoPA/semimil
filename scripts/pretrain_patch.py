#!/usr/bin/env python
"""Patch-level domain adaptation of DINOv2: anchored masked-token distillation.

The paper's DINO adaptation (``pretrain_ssl.py``) degraded the detector, and we
attributed that to two things: DINO supervises the *global* CLS token while the
detector consumes *patch* tokens, and its aggressive augmentations (color jitter,
blur, solarization, rescaled crops) wash out the fine detail anomaly detection needs.
This script tests the fix directly:

* **Patch-level objective.**  The student sees an IC image with a random subset of
  patch tokens replaced by a learned mask token and must predict, at the masked
  positions, the patch tokens that an EMA teacher computes on the full image
  (data2vec/iBOT-style masked feature prediction; targets are the layer-normalized
  average of the teacher's last ``target_layers`` blocks).  Predicting a token from
  its context only works if the token encodes the regular structure of normal IC
  layouts, which is exactly what makes a defect stand out in the memory bank.
* **Anchor.**  Visible tokens are tied to the frozen, original DINOv2 tokens
  (cosine loss, weight ``anchor_weight``), so adaptation cannot drift into a
  representation that forgets what made DINOv2 work.
* **Gentle views.**  Native-scale random crops with flips and 90-degree rotations
  only: these are symmetries of IC layouts and keep every pixel's contrast intact.
* **Partial fine-tuning.**  Only the last ``train_blocks`` transformer blocks (plus
  the final norm and the mask token) are trained.

Saves the student backbone as a timm state dict loadable by ``run_patchcore`` /
``run_tiled`` via ``model.weights``.  Argument-free: reads ``configs/patch_ssl.yaml``
(or the YAML named by ``SEMIMIL_CONFIG``).
"""

from __future__ import annotations

import copy
import json
import math
import os
import random
import time
from pathlib import Path

import certifi
import numpy as np
import timm
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from semimil.utils import seed_everything

os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(os.environ.get("SEMIMIL_CONFIG", REPO_ROOT / "configs" / "patch_ssl.yaml"))
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


# ----------------------------------------------------------------------------- #
# data
# ----------------------------------------------------------------------------- #
class NormalCrops(Dataset):
    """Native-scale random crops of normal IC images with D4 symmetries only."""

    def __init__(self, paths: list[Path], crop: int):
        self.paths = paths
        self.crop = crop

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> torch.Tensor:
        im = np.asarray(Image.open(self.paths[i]).convert("L"), np.float32) / 255.0
        h, w = im.shape
        y = random.randint(0, h - self.crop)
        x = random.randint(0, w - self.crop)
        im = im[y : y + self.crop, x : x + self.crop]
        im = np.rot90(im, random.randint(0, 3))
        if random.random() < 0.5:
            im = im[:, ::-1]
        im = np.repeat(np.ascontiguousarray(im)[None], 3, 0)
        return torch.from_numpy((im - MEAN[:, None, None]) / STD[:, None, None])


# ----------------------------------------------------------------------------- #
# masked forward through a timm ViT
# ----------------------------------------------------------------------------- #
def forward_tokens(vit, x, mask=None, mask_token=None, keep_layers=1):
    """Patch tokens of a timm ViT, optionally with masked patches replaced.

    Returns the final-norm patch tokens [B, N, D] and the list of the last
    ``keep_layers`` block outputs (patch tokens only, before the final norm).
    """
    t = vit.patch_embed(x)  # [B, H, W, D] with dynamic_img_size
    b, h, w, d = t.shape
    if mask is not None:
        m = mask.view(b, h, w, 1).to(t.dtype)
        t = t * (1 - m) + mask_token.to(t.dtype).view(1, 1, 1, d) * m
    t = vit._pos_embed(t)
    t = vit.patch_drop(t)
    t = vit.norm_pre(t)
    p = vit.num_prefix_tokens
    outs = []
    n_blocks = len(vit.blocks)
    for i, blk in enumerate(vit.blocks):
        t = blk(t)
        if i >= n_blocks - keep_layers:
            outs.append(t[:, p:])
    return vit.norm(t)[:, p:], outs


def random_mask(b, n, ratio, device, gen=None):
    """Boolean [B, N] mask with round(ratio * N) masked patches per image."""
    k = int(round(ratio * n))
    noise = torch.rand(b, n, device=device, generator=gen)
    idx = noise.argsort(dim=1)[:, :k]
    mask = torch.zeros(b, n, dtype=torch.bool, device=device)
    mask.scatter_(1, idx, True)
    return mask


def cosine_sched(step, total, base, final):
    return final + 0.5 * (base - final) * (1 + math.cos(math.pi * step / max(1, total)))


def main() -> None:
    cfg = yaml.safe_load(CONFIG_PATH.open())
    seed_everything(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tc, mc = cfg["train"], cfg["model"]
    out_dir = Path(cfg["output"]["dir"]) / cfg["run_name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = Path(cfg["data"]["raw_dir"])
    paths = [p for p in sorted(raw.glob(cfg["data"]["train_normal_glob"]))
             if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if cfg["data"]["limit_images"]:
        paths = paths[: cfg["data"]["limit_images"]]
    ds = NormalCrops(paths, tc["crop"])
    dl = DataLoader(ds, batch_size=tc["batch_size"], shuffle=True, drop_last=True,
                    num_workers=cfg["data"]["num_workers"], pin_memory=True,
                    persistent_workers=cfg["data"]["num_workers"] > 0)

    student = timm.create_model(mc["dino_name"], pretrained=True, num_classes=0,
                                dynamic_img_size=True).to(device)
    anchor = copy.deepcopy(student).eval()
    teacher = copy.deepcopy(student).eval()
    for m in (anchor, teacher):
        for q in m.parameters():
            q.requires_grad_(False)

    # train only the last blocks + final norm; everything else stays DINOv2
    for q in student.parameters():
        q.requires_grad_(False)
    n_blocks = len(student.blocks)
    trainable = list(student.blocks[n_blocks - tc["train_blocks"] :]) + [student.norm]
    for mod in trainable:
        for q in mod.parameters():
            q.requires_grad_(True)
    d = student.embed_dim
    mask_token = nn.Parameter(torch.zeros(d, device=device))
    nn.init.normal_(mask_token, std=0.02)
    # predictor: small MLP from student tokens to the teacher target space
    predictor = nn.Sequential(nn.Linear(d, d * 2), nn.GELU(), nn.Linear(d * 2, d)).to(device)
    params = [q for q in student.parameters() if q.requires_grad]
    params += [mask_token] + list(predictor.parameters())
    n_train = sum(q.numel() for q in params)
    opt = torch.optim.AdamW(params, lr=tc["lr"], weight_decay=tc["weight_decay"])

    steps_per_epoch = len(dl)
    total = tc["epochs"] * steps_per_epoch
    warmup = tc["warmup_epochs"] * steps_per_epoch
    print(f"device={device} images={len(paths)} crop={tc['crop']} steps/epoch={steps_per_epoch} "
          f"total={total} trainable={n_train / 1e6:.1f}M (last {tc['train_blocks']} blocks) "
          f"mask={tc['mask_ratio']} anchor={tc['anchor_weight']}", flush=True)

    student.train()
    log = []
    gstep = 0
    for epoch in range(tc["epochs"]):
        t0 = time.time()
        run = {"loss": 0.0, "mim": 0.0, "anchor": 0.0}
        for x in dl:
            x = x.to(device, non_blocking=True)
            lr = (tc["lr"] * gstep / max(1, warmup) if gstep < warmup
                  else cosine_sched(gstep - warmup, total - warmup, tc["lr"], tc["min_lr"]))
            for g in opt.param_groups:
                g["lr"] = lr
            n = (x.shape[-1] // 14) ** 2
            mask = random_mask(x.shape[0], n, tc["mask_ratio"], device)
            with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
                with torch.no_grad():
                    _, t_layers = forward_tokens(teacher, x, keep_layers=tc["target_layers"])
                    target = torch.stack(
                        [F.layer_norm(z.float(), z.shape[-1:]) for z in t_layers]
                    ).mean(0)
                    target = F.layer_norm(target, target.shape[-1:])
                    a_tok, _ = forward_tokens(anchor, x)
                s_tok, _ = forward_tokens(student, x, mask=mask, mask_token=mask_token)
                pred = predictor(s_tok)
                mim = F.smooth_l1_loss(pred[mask].float(), target[mask], beta=tc["beta"])
                vis = ~mask
                anc = (1 - F.cosine_similarity(s_tok[vis].float(), a_tok[vis].float(), dim=-1)).mean()
                loss = mim + tc["anchor_weight"] * anc
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 3.0)
            opt.step()
            mom = cosine_sched(gstep, total, tc["ema_momentum"], 1.0)
            with torch.no_grad():
                for ps, pt in zip(student.parameters(), teacher.parameters(), strict=True):
                    pt.mul_(mom).add_(ps.detach(), alpha=1 - mom)
            run["loss"] += loss.item()
            run["mim"] += mim.item()
            run["anchor"] += anc.item()
            gstep += 1
        rec = {k: v / steps_per_epoch for k, v in run.items()}
        rec.update(epoch=epoch, lr=lr, sec=time.time() - t0)
        log.append(rec)
        print(f"epoch {epoch:2d} loss {rec['loss']:.4f} mim {rec['mim']:.4f} "
              f"anchor {rec['anchor']:.4f} lr {lr:.2e} ({rec['sec']:.0f}s)", flush=True)
        torch.save(student.state_dict(), out_dir / "backbone.pth")
        (out_dir / "train_log.json").write_text(json.dumps(log, indent=2))
    print(f"saved -> {out_dir / 'backbone.pth'}", flush=True)


if __name__ == "__main__":
    main()
