#!/usr/bin/env python
"""GLASS (ECCV 2024) on MIIC: official custom-dataset recipe (shell/run-custom.sh).

Reproduces ``main.py`` (net + dataset commands, run callback) with the repo's own modules and the
exact run-custom.sh flags: WideResNet-50 layer2+layer3, embed 1536, patchsize 3, 640 meta-epochs,
discriminator 2x1024, pre-projection, gradient-ascent mining (noise 0.015, radius 0.75, p 0.5,
step 20, limit 392), --distribution 2 (manifold), --fg 0, rand_aug 1, DTD textures for local
Perlin synthesis (mean 0.5, std 0.1), batch 8, resize/imagesize 288.

Leakage control: the official trainer evaluates on the test loader after every epoch and keeps
the checkpoint with the best test I-AUROC + P-AUROC (test-set checkpoint selection).  Here
eval_epochs is set beyond meta_epochs, no test loader is passed to the trainer, and the final
(epoch-640) weights are scored.  Image score / map exactly as ``GLASS._predict``.

Deviations: the trainer recomputes the feature centre over the whole training set (with a fresh
DTD/Perlin augmentation per image) every epoch, so 640 epochs on all 23,888 normals would be
~15M augmented loads (~2 days/seed on 3 CPUs); a seeded random subsample of N_TRAIN=1,000 normals
is used (same bank size as the paper's detector).  num_workers 16 -> 3 (cluster CPU limit).

Env vars: SEED (0), N_TRAIN (1000), META_EPOCHS (640), SETTING (default "custom_ep640_n1000").
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

REPO = Path("/autofs/space/crater_001/projects/micropath/semi/external_code/glass")
DTD = "/autofs/space/crater_001/projects/micropath/semi/external_code/glass_data/dtd/images"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sota_common as sc  # noqa: E402

sys.path.insert(0, str(REPO))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import backbones  # noqa: E402
import glass  # noqa: E402
import utils  # noqa: E402
from datasets import mvtec  # noqa: E402


def main() -> None:
    seed = int(os.environ.get("SEED", 0))
    n_train = int(os.environ.get("N_TRAIN", 1000))
    meta_epochs = int(os.environ.get("META_EPOCHS", 640))
    setting = os.environ.get("SETTING", f"custom_ep{meta_epochs}_n{n_train}")
    out = sc.run_dir("glass", setting, seed)
    work = out / "glass_work"
    work.mkdir(exist_ok=True)
    os.chdir(work)   # the repo writes ./results/... relative to the CWD

    # ---- dataset command (run-custom.sh flags) ----
    ds_kw = dict(resize=288, imagesize=288)
    train_dataset = mvtec.MVTecDataset(
        str(sc.TREE), DTD, dataset_name="mvtec", classname=sc.CAT, split=mvtec.DatasetSplit.TRAIN,
        seed=seed, rotate_degrees=0, translate=0, brightness_factor=0, contrast_factor=0,
        saturation_factor=0, gray_p=0, h_flip_p=0, v_flip_p=0, scale=0.0, distribution=2,
        mean=0.5, std=0.1, fg=0, rand_aug=1, downsampling=8, augment=False, batch_size=8, **ds_kw)
    keep = {str(p) for p in sc.subsample(sc.train_paths(), n_train, seed)}
    train_dataset.data_to_iterate = [t for t in train_dataset.data_to_iterate if t[2] in keep]
    assert len(train_dataset) == n_train, len(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=3,
                                               prefetch_factor=2, pin_memory=True)
    train_loader.name = "mvtec_" + sc.CAT
    test_dataset = mvtec.MVTecDataset(str(sc.TREE), DTD, classname=sc.CAT, split=mvtec.DatasetSplit.TEST,
                                      seed=seed, **ds_kw)

    # ---- run callback ----
    device = utils.set_torch_device([0])
    utils.fix_seeds(seed, device)
    backbone = backbones.load("wideresnet50")
    backbone.name, backbone.seed = "wideresnet50", None
    model = glass.GLASS(device)
    model.load(backbone=backbone, layers_to_extract_from=["layer2", "layer3"], device=device,
               input_shape=train_dataset.imagesize, pretrain_embed_dimension=1536,
               target_embed_dimension=1536, patchsize=3, meta_epochs=meta_epochs,
               eval_epochs=10 ** 9,                        # never evaluate on the test set
               dsc_layers=2, dsc_hidden=1024, dsc_margin=0.5, train_backbone=False, pre_proj=1,
               mining=1, noise=0.015, radius=0.75, p=0.5, lr=0.0001, svd=0, step=20, limit=392)
    model.to(device)
    model.set_model_dir(str(work / "models" / "backbone_0"), train_loader.name)
    print(f"seed={seed} train={len(train_dataset)} test={len(test_dataset)} epochs={meta_epochs}", flush=True)

    t0 = time.time()
    rec = model.trainer(train_loader, None, train_loader.name)   # no validation data -> final epoch
    assert rec is None, rec
    train_time = time.time() - t0

    # ---- inference with the final weights ----
    t1 = time.time()
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=3)
    by_name = {}
    for data in test_loader:
        scores, masks = model._predict(data["image"])
        for p, y, s, m in zip(data["image_path"], data["is_anomaly"].numpy(), scores, masks):
            by_name[sc.original_name(p)] = (int(y), float(s), m)
    infer_time = time.time() - t1
    items = sc.test_items()
    names = [sc.original_name(p) for p, _ in items]
    labels = [by_name[n][0] for n in names]
    scores = [by_name[n][1] for n in names]
    maps = sc.resize_maps([by_name[n][2] for n in names])
    sc.save_run(out, names, labels, scores, maps, {
        "method": "GLASS", "repo_commit": "f788d56", "setting": setting, "seed": seed,
        "n_train_images": n_train, "meta_epochs": meta_epochs, "limit": 392, "batch_size": 8,
        "resize": 288, "distribution": 2, "fg": 0, "svd": int(model.svd),
        "train_time_s": train_time, "inference_time_s": infer_time,
        "deviations": [f"seeded random subsample of {n_train} training normals (per-epoch full-set centre pass)",
                       "no per-epoch test evaluation / best-checkpoint selection (final epoch scored)",
                       "num_workers 3"]})


if __name__ == "__main__":
    main()
