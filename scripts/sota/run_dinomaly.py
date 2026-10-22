#!/usr/bin/env python
"""Dinomaly (CVPR 2025) on MIIC: official class-separated recipe (dinomaly_mvtec_sep.py).

Reproduces ``train()`` of the official script with the repo's own modules: frozen DINOv2-reg ViT-B/14
encoder, noisy MLP bottleneck (dropout 0.2), 8 linear-attention decoder blocks, loose grouped
cosine loss with hard-mining ramp, StableAdamW 2e-3 + warm-cosine, 5,000 iterations, batch 16.
The final iterate is scored (no test-set checkpoint selection; the official script also
evaluates only at iteration 5,000).  Anomaly map / image score exactly as
``utils.evaluation_batch(max_ratio=0.01, resize_mask=256)``.

Deviation: the official R448-C392 center crop would discard the outer 6 % border, where 22 of
the 116 MIIC defects mostly lie, so images are resized to 448x448 without cropping (same
resize, crop disabled).  num_workers 4 -> 3 (cluster CPU limit).

Env vars: SEED (0), TRAIN_ROOT (MVTec-style root holding miic/train; default the symlink tree),
          SETTING (default "sep_it5000_r448").
"""

from __future__ import annotations

import os
import sys
import time
from functools import partial
from pathlib import Path

REPO = Path("/autofs/space/crater_001/projects/micropath/semi/external_code/dinomaly")
WORK = Path("/autofs/space/crater_001/projects/micropath/semi/external_code/work/dinomaly")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sota_common as sc  # noqa: E402

WORK.mkdir(parents=True, exist_ok=True)
os.chdir(WORK)                      # vit_encoder caches weights in ./backbones/weights
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402
from torchvision.datasets import ImageFolder  # noqa: E402

from dataset import get_data_transforms  # noqa: E402
from dinov1.utils import trunc_normal_  # noqa: E402
from models import vit_encoder  # noqa: E402
from models.uad import ViTill  # noqa: E402
from models.vision_transformer import Block as VitBlock, bMlp, LinearAttention2  # noqa: E402
from optimizers import StableAdamW  # noqa: E402
from utils import cal_anomaly_maps, get_gaussian_kernel, global_cosine_hm_percent, WarmCosineScheduler  # noqa: E402


def setup_seed(seed):  # copied from dinomaly_mvtec_sep.py
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main() -> None:
    seed = int(os.environ.get("SEED", 0))
    setting = os.environ.get("SETTING", "sep_it5000_r448")
    train_root = Path(os.environ.get("TRAIN_ROOT", sc.TREE))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    setup_seed(seed)

    total_iters, batch_size, image_size, crop_size = 5000, 16, 448, 448   # official crop 392 disabled
    total_iters = int(os.environ.get("TOTAL_ITERS", total_iters))       # smoke tests only
    data_transform, _ = get_data_transforms(image_size, crop_size)
    train_data = ImageFolder(root=str(train_root / sc.CAT / "train"), transform=data_transform)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                                   num_workers=3, drop_last=True)

    encoder_name = "dinov2reg_vit_base_14"
    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    encoder = vit_encoder.load(encoder_name)
    embed_dim, num_heads = 768, 12

    bottleneck = nn.ModuleList([bMlp(embed_dim, embed_dim * 4, embed_dim, drop=0.2)])
    decoder = nn.ModuleList([
        VitBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=4., qkv_bias=True,
                 norm_layer=partial(nn.LayerNorm, eps=1e-8), attn_drop=0., attn=LinearAttention2)
        for _ in range(8)])
    model = ViTill(encoder=encoder, bottleneck=bottleneck, decoder=decoder, target_layers=target_layers,
                   mask_neighbor_size=0, fuse_layer_encoder=fuse_layer_encoder,
                   fuse_layer_decoder=fuse_layer_decoder).to(device)
    trainable = nn.ModuleList([bottleneck, decoder])
    for m in trainable.modules():
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    optimizer = StableAdamW([{"params": trainable.parameters()}], lr=2e-3, betas=(0.9, 0.999),
                            weight_decay=1e-4, amsgrad=True, eps=1e-8)
    lr_scheduler = WarmCosineScheduler(optimizer, base_value=2e-3, final_value=2e-4,
                                       total_iters=total_iters, warmup_iters=100)
    print(f"seed={seed} train images={len(train_data)} root={train_root}", flush=True)

    t0 = time.time()
    it = 0
    for epoch in range(int(np.ceil(total_iters / len(train_dataloader)))):
        model.train()
        loss_list = []
        for img, _ in train_dataloader:
            img = img.to(device)
            en, de = model(img)
            p_final = 0.9
            p = min(p_final * it / 1000, p_final)
            loss = global_cosine_hm_percent(en, de, p=p, factor=0.1)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable.parameters(), max_norm=0.1)
            optimizer.step()
            loss_list.append(loss.item())
            lr_scheduler.step()
            it += 1
            if it % 100 == 0:
                print(f"iter [{it}/{total_iters}] loss:{np.mean(loss_list):.4f} "
                      f"({time.time() - t0:.0f}s)", flush=True)
                loss_list = []
            if it == total_iters:
                break
        if it == total_iters:
            break
    train_time = time.time() - t0

    # ---- inference on the full real test split (final iterate) ----
    t1 = time.time()
    model.eval()
    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    items = sc.test_items()
    names, labels, scores, maps = [], [], [], []
    with torch.no_grad():
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            img = torch.stack([data_transform(Image.open(p).convert("RGB")) for p, _ in batch]).to(device)
            en, de = model(img)
            amap, _ = cal_anomaly_maps(en, de, img.shape[-1])
            amap = F.interpolate(amap, size=256, mode="bilinear", align_corners=False)
            amap = gaussian_kernel(amap)
            flat = amap.flatten(1)
            sp = torch.sort(flat, dim=1, descending=True)[0][:, :int(flat.shape[1] * 0.01)].mean(dim=1)
            for (p, y), s, m in zip(batch, sp.cpu().numpy(), amap[:, 0].cpu().numpy()):
                names.append(sc.original_name(p)); labels.append(y); scores.append(float(s)); maps.append(m)
    infer_time = time.time() - t1
    d = sc.run_dir("dinomaly", setting, seed)
    sc.save_run(d, names, labels, scores, np.stack(maps).astype(np.float32), {
        "method": "Dinomaly", "repo_commit": "1f252be", "setting": setting, "seed": seed,
        "n_train_images": len(train_data), "total_iters": total_iters, "batch_size": batch_size,
        "resize": image_size, "crop": None, "encoder": encoder_name,
        "train_time_s": train_time, "inference_time_s": infer_time,
        "deviations": ["no center crop (R448 instead of R448-C392)", "num_workers 3"]})


if __name__ == "__main__":
    main()
