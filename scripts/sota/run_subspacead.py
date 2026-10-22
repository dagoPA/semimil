#!/usr/bin/env python
"""SubspaceAD (CVPR 2026) on MIIC: training-free PCA subspace over frozen DINOv2 patch features.

Uses the repo's own components (FeatureExtractor, streaming two-pass PCAModel, reconstruction
scoring, post_process_map, mtop1p image score, rotation augmentation) with the defaults of
scripts/benchmark_few_shot.sh and scripts/benchmark_full_shot.sh:
    facebook/dinov2-with-registers-giant, image_res 672, layers -12..-18 (mean), pca_ev 0.99,
    reconstruction residual, mtop1p image score, no crop, no masking, batch_size 1.
    few-shot : k in {1,2,4} random normals (seeded shuffle, as main.py) + 30 random rotations each
    full-shot: all 23,888 training normals, no augmentation (deterministic, one run)
The official main.py evaluation keeps every test pixel in Python lists (~75 GB for 1,388 images at
672x672), so the per-image loop is re-implemented here with the same calls; nothing is fit on
test images.  Pixel maps are the official 672x672 post-processed maps, resized to 256x256.

Env vars: MODE ("few" | "full" | "both"), SHOTS ("1,2,4"), SEEDS ("0,1,2").
"""

from __future__ import annotations

import math
import os
import random
import sys
import time
from pathlib import Path

REPO = Path("/autofs/space/crater_001/projects/micropath/semi/external_code/subspacead")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sota_common as sc  # noqa: E402

sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from src.subspacead.config import parse_layer_indices  # noqa: E402
from src.subspacead.core.extractor import FeatureExtractor  # noqa: E402
from src.subspacead.core.pca import PCAModel  # noqa: E402
from src.subspacead.data.transforms import get_augmentation_transform  # noqa: E402
from src.subspacead.post_process.scoring import calculate_anomaly_scores, post_process_map  # noqa: E402
from src.subspacead.utils.common import topk_mean  # noqa: E402

CKPT = "facebook/dinov2-with-registers-giant"
RES, LAYERS, AGG, PCA_EV, AUG_COUNT, BATCH = 672, "-12,-13,-14,-15,-16,-17,-18", "mean", 0.99, 30, 1
SAL_LAYER = 6  # config default (only used for the unused saliency output)


def seed_all(seed: int) -> None:  # as main.py
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def run_one(extractor, k: int | None, seed: int) -> None:
    layers = parse_layer_indices(LAYERS)
    setting = f"fewshot_k{k}_aug{AUG_COUNT}" if k else "fullshot_all"
    seed_all(seed)
    train = [str(p) for p in sc.train_paths()]
    aug = None
    if k is not None:
        aug = get_augmentation_transform(["rotate"], RES)
        random.shuffle(train)
        train = train[:k]
        print(f"[{setting} seed{seed}] k-shot images: {[Path(p).name for p in train]}", flush=True)
    elif os.environ.get("TRAIN_LIMIT"):                                    # smoke tests only
        train = train[:int(os.environ["TRAIN_LIMIT"])]

    def ext(imgs):
        return extractor.extract_tokens(imgs, RES, layers, AGG, [], False, use_clahe=False,
                                        dino_saliency_layer=SAL_LAYER)

    tmp, (h_p, w_p), _ = ext([Image.open(train[0]).convert("RGB")])
    feature_dim = tmp.shape[-1]
    mult = (1 + AUG_COUNT) if aug else 1
    total_imgs = len(train) * mult
    total_tokens = total_imgs * h_p * w_p
    num_batches = math.ceil(total_imgs / BATCH)

    def feature_generator():
        if aug is not None:   # official feature_generator_full: originals + fresh augmentations
            imgs = []
            for p in train:
                pil = Image.open(p).convert("RGB")
                imgs.append(pil)
                imgs.extend(aug(pil) for _ in range(AUG_COUNT))
            for i in range(0, len(imgs), BATCH):
                tok, _, _ = ext(imgs[i:i + BATCH])
                yield tok.reshape(-1, feature_dim)
        else:                 # same order/batches, streamed instead of holding 23,888 PIL images
            for i in range(0, len(train), BATCH):
                tok, _, _ = ext([Image.open(p).convert("RGB") for p in train[i:i + BATCH]])
                yield tok.reshape(-1, feature_dim)

    t0 = time.time()
    pca = PCAModel(k=None, ev=PCA_EV, whiten=False).fit(feature_generator, feature_dim, total_tokens, num_batches)
    fit_time = time.time() - t0
    print(f"[{setting} seed{seed}] PCA k={pca['k']} fit {fit_time:.0f}s on {total_imgs} images", flush=True)

    t1 = time.time()
    names, labels, scores, maps = [], [], [], []
    items = sc.test_items()
    for i in range(0, len(items), BATCH):
        batch = items[i:i + BATCH]
        pil_imgs = [Image.open(p).convert("RGB") for p, _ in batch]
        tok, (h_p, w_p), _ = ext(pil_imgs)
        b, _, _, c = tok.shape
        s = calculate_anomaly_scores(tok.reshape(b * h_p * w_p, c), pca, "reconstruction", 0)
        amaps = s.reshape(b, h_p, w_p)
        for j, (p, y) in enumerate(batch):
            final = post_process_map(amaps[j], RES)
            names.append(sc.original_name(p)); labels.append(y)
            scores.append(topk_mean(final, frac=0.01)); maps.append(final)
        if (i // BATCH) % 200 == 0:
            print(f"  test {i}/{len(items)} ({time.time() - t1:.0f}s)", flush=True)
    infer_time = time.time() - t1
    d = sc.run_dir("subspacead", setting, seed)
    sc.save_run(d, names, labels, scores, sc.resize_maps(maps), {
        "method": "SubspaceAD", "repo_commit": "ef56d5c", "setting": setting, "seed": seed,
        "k_shot": k, "aug_count": AUG_COUNT if k else 0, "n_train_images": len(train),
        "model_ckpt": CKPT, "image_res": RES, "layers": LAYERS, "agg": AGG, "pca_ev": PCA_EV,
        "pca_k": int(pca["k"]), "train_time_s": fit_time, "inference_time_s": infer_time,
        "deviations": ["own per-image evaluation loop (official keeps all pixels in Python lists)",
                       "seeds 0,1,2 instead of 42" if k else "full-shot streamed (no 19 GB image list)"]})


def main() -> None:
    mode = os.environ.get("MODE", "few")
    extractor = FeatureExtractor(CKPT)
    if mode in ("few", "both"):
        for k in [int(x) for x in os.environ.get("SHOTS", "1,2,4").split(",")]:
            for seed in [int(x) for x in os.environ.get("SEEDS", "0,1,2").split(",")]:
                run_one(extractor, k, seed)
    if mode in ("full", "both"):
        run_one(extractor, None, 0)


if __name__ == "__main__":
    main()
