#!/usr/bin/env python
"""INP-Former (CVPR 2025) on MIIC: official single-class recipe (INP_Former_Single_Class.py).

Reproduces ``main()`` of the official script with the repo's own modules: frozen DINOv2-reg
ViT-B/14, 6 intrinsic normal prototypes (INP extractor = 1 aggregation block), MLP bottleneck,
8 INP-guided decoder blocks, soft-mining cosine loss (y=3) + 0.2 * INP coherence loss,
StableAdamW 1e-3 + warm-cosine, 200 epochs, batch 16.  The final model is scored (the official
script also evaluates once, after the last epoch).  Anomaly map / image score exactly as
``utils.evaluation_batch(max_ratio=0.01, resize_mask=256)``.

Deviations (both stated in run_info.json):
  * budget: 200 epochs over all 23,888 normals would be 298,600 iterations (~1-2 GPU-days per
    seed); the official budget is epoch-based, so a seeded random subsample of N_TRAIN=1,000
    normals is used (same bank size as the paper's detector): 200 epochs = 12,400 iterations.
  * no center crop (R448 instead of R448-C392): the crop discards the border where 22/116
    MIIC defects mostly lie.  num_workers 4 -> 3 (cluster CPU limit).

Env vars: SEED (0), N_TRAIN (1000), SETTING (default "single_ep200_n1000_r448").
"""

from __future__ import annotations

import os
import sys
import time
from functools import partial
from pathlib import Path

REPO = Path("/autofs/space/crater_001/projects/micropath/semi/external_code/inp_former")
WORK = Path("/autofs/space/crater_001/projects/micropath/semi/external_code/work/inp_former")
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
from torch.nn.init import trunc_normal_  # noqa: E402
from torchvision.datasets import ImageFolder  # noqa: E402

from dataset import get_data_transforms  # noqa: E402
from models import vit_encoder  # noqa: E402
from models.uad import INP_Former  # noqa: E402
from models.vision_transformer import Mlp, Aggregation_Block, Prototype_Block  # noqa: E402
from optimizers import StableAdamW  # noqa: E402
from utils import (WarmCosineScheduler, cal_anomaly_maps, get_gaussian_kernel,  # noqa: E402
                   global_cosine_hm_adaptive, setup_seed)


def main() -> None:
    seed = int(os.environ.get("SEED", 0))
    n_train = int(os.environ.get("N_TRAIN", 1000))
    setting = os.environ.get("SETTING", f"single_ep200_n{n_train}_r448")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    setup_seed(seed)

    input_size, crop_size, inp_num, total_epochs, batch_size = 448, 448, 6, 200, 16
    total_epochs = int(os.environ.get("EPOCHS", total_epochs))          # smoke tests only
    encoder_name = "dinov2reg_vit_base_14"
    data_transform, _ = get_data_transforms(input_size, crop_size)
    full = ImageFolder(root=str(sc.TREE / sc.CAT / "train"), transform=data_transform)
    all_paths = [Path(p) for p, _ in full.samples]
    keep = set(map(str, sc.subsample(all_paths, n_train, seed)))
    idx = [i for i, (p, _) in enumerate(full.samples) if p in keep]
    train_data = torch.utils.data.Subset(full, idx)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                                   num_workers=3, drop_last=True)

    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    encoder = vit_encoder.load(encoder_name)
    embed_dim, num_heads = 768, 12

    Bottleneck = nn.ModuleList([Mlp(embed_dim, embed_dim * 4, embed_dim, drop=0.)])
    INP = nn.ParameterList([nn.Parameter(torch.randn(inp_num, embed_dim)) for _ in range(1)])
    INP_Extractor = nn.ModuleList([
        Aggregation_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4., qkv_bias=True,
                          norm_layer=partial(nn.LayerNorm, eps=1e-8)) for _ in range(1)])
    INP_Guided_Decoder = nn.ModuleList([
        Prototype_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4., qkv_bias=True,
                        norm_layer=partial(nn.LayerNorm, eps=1e-8)) for _ in range(8)])
    model = INP_Former(encoder=encoder, bottleneck=Bottleneck, aggregation=INP_Extractor,
                       decoder=INP_Guided_Decoder, target_layers=target_layers, remove_class_token=True,
                       fuse_layer_encoder=fuse_layer_encoder, fuse_layer_decoder=fuse_layer_decoder,
                       prototype_token=INP).to(device)

    trainable = nn.ModuleList([Bottleneck, INP_Guided_Decoder, INP_Extractor, INP])
    for m in trainable.modules():
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    optimizer = StableAdamW([{"params": trainable.parameters()}], lr=1e-3, betas=(0.9, 0.999),
                            weight_decay=1e-4, amsgrad=True, eps=1e-10)
    lr_scheduler = WarmCosineScheduler(optimizer, base_value=1e-3, final_value=1e-4,
                                       total_iters=total_epochs * len(train_dataloader), warmup_iters=100)
    print(f"seed={seed} train images={len(train_data)} iters/epoch={len(train_dataloader)}", flush=True)

    t0 = time.time()
    for epoch in range(total_epochs):
        model.train()
        loss_list = []
        for img, _ in train_dataloader:
            img = img.to(device)
            en, de, g_loss = model(img)
            loss = global_cosine_hm_adaptive(en, de, y=3)
            loss = loss + 0.2 * g_loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable.parameters(), max_norm=0.1)
            optimizer.step()
            loss_list.append(loss.item())
            lr_scheduler.step()
        print(f"epoch [{epoch + 1}/{total_epochs}], loss:{np.mean(loss_list):.4f} ({time.time() - t0:.0f}s)",
              flush=True)
    train_time = time.time() - t0
    torch.save({k: v for k, v in model.state_dict().items() if not k.startswith("encoder.")},
               sc.run_dir("inp_former", setting, seed) / "trainable_final.pth")

    # ---- inference on the full real test split (final model) ----
    t1 = time.time()
    model.eval()
    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    items = sc.test_items()
    names, labels, scores, maps = [], [], [], []
    with torch.no_grad():
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            img = torch.stack([data_transform(Image.open(p).convert("RGB")) for p, _ in batch]).to(device)
            output = model(img)
            en, de = output[0], output[1]
            amap, _ = cal_anomaly_maps(en, de, img.shape[-1])
            amap = F.interpolate(amap, size=256, mode="bilinear", align_corners=False)
            amap = gaussian_kernel(amap)
            flat = amap.flatten(1)
            sp = torch.sort(flat, dim=1, descending=True)[0][:, :int(flat.shape[1] * 0.01)].mean(dim=1)
            for (p, y), s, m in zip(batch, sp.cpu().numpy(), amap[:, 0].cpu().numpy()):
                names.append(sc.original_name(p)); labels.append(y); scores.append(float(s)); maps.append(m)
    infer_time = time.time() - t1
    d = sc.run_dir("inp_former", setting, seed)
    sc.save_run(d, names, labels, scores, np.stack(maps).astype(np.float32), {
        "method": "INP-Former", "repo_commit": "17d2653", "setting": setting, "seed": seed,
        "n_train_images": len(train_data), "epochs": total_epochs,
        "iterations": total_epochs * len(train_dataloader), "batch_size": batch_size,
        "resize": input_size, "crop": None, "encoder": encoder_name, "INP_num": inp_num,
        "train_time_s": train_time, "inference_time_s": infer_time,
        "deviations": [f"seeded random subsample of {n_train} training normals (epoch-based budget)",
                       "no center crop (R448 instead of R448-C392)", "num_workers 3"]})


if __name__ == "__main__":
    main()
