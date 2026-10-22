#!/usr/bin/env python
"""AnomalyDINO (WACV 2025) on MIIC: DINOv2 ViT-S/14 patch 1-NN, official run_anomaly_detection().

Calls the repo's ``src.detection.run_anomaly_detection`` per (shot, seed) exactly as
run_anomalydino.py does, with the README defaults: dinov2_vits14, resolution 448, L2-normalized
(cosine) 1-NN via faiss-GPU, mean-of-top-1 % patch distances as image score; preprocessing from
``get_dataset_info`` for a custom dataset ("agnostic_no_mask": no masking, 8 rotations of every
reference image).  Few-shot references are the official deterministic slice
sorted(train/good)[seed*k:(seed+1)*k].  Pixel maps: official dists2map (bilinear to 512x512 +
Gaussian sigma 4) resized to 256x256.

Env vars: SHOTS ("1,2,4,8,16"; -1 = full-shot), SEEDS ("0,1,2").
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

REPO = Path("/autofs/space/crater_001/projects/micropath/semi/external_code/anomalydino")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sota_common as sc  # noqa: E402

sys.path.insert(0, str(REPO))
import numpy as np  # noqa: E402

from src.backbones import get_model  # noqa: E402
from src.detection import run_anomaly_detection  # noqa: E402
from src.utils import dists2map, get_dataset_info  # noqa: E402


def main() -> None:
    shots = [int(x) for x in os.environ.get("SHOTS", "1,2,4,8,16").split(",")]
    seeds = [int(x) for x in os.environ.get("SEEDS", "0,1,2").split(",")]
    objects, object_anomalies, masking_default, rotation_default = get_dataset_info(
        "MIIC", "agnostic", data_path=str(sc.TREE))
    assert objects == [sc.CAT], objects
    print("masking:", masking_default, "rotation:", rotation_default, flush=True)
    model = get_model("dinov2_vits14", "cuda", smaller_edge_size=448)
    for shot in shots:
        setting = f"{shot}shot" if shot > 0 else "fullshot"
        for seed in seeds:
            d = sc.run_dir("anomalydino", setting, seed)
            t0 = time.time()
            scores, t_mem, t_inf = run_anomaly_detection(
                model, sc.CAT, data_root=str(sc.TREE), n_ref_samples=shot,
                object_anomalies=object_anomalies, plots_dir=str(d / "raw"), save_examples=False,
                knn_metric="L2_normalized", knn_neighbors=1, faiss_on_cpu=False,
                masking=masking_default[sc.CAT], mask_ref_images=False,
                rotation=rotation_default[sc.CAT], seed=seed, save_patch_dists=True, save_tiffs=False)
            total = time.time() - t0
            names, labels, sc_list, maps = [], [], [], []
            for p, y in sc.test_items():
                typ = "good" if y == 0 else "defect"
                dist = np.load(d / "raw" / "anomaly_maps" / f"seed={seed}" / sc.CAT / "test" / typ / f"{p.stem}.npy")
                names.append(sc.original_name(p)); labels.append(y)
                sc_list.append(float(scores[f"{typ}/{p.name}"]))
                maps.append(dists2map(dist, (512, 512)))
            n_ref = len(sc.train_paths()) if shot == -1 else min(shot, len(sc.train_paths()) - seed * shot)
            sc.save_run(d, names, labels, sc_list, sc.resize_maps(maps), {
                "method": "AnomalyDINO", "repo_commit": "b9d1c26", "setting": setting, "seed": seed,
                "shots": shot, "n_train_images": n_ref, "reference_slice": [seed * shot, (seed + 1) * shot],
                "model": "dinov2_vits14", "resolution": 448, "preprocess": "agnostic_no_mask (custom dataset)",
                "rotation": rotation_default[sc.CAT], "masking": masking_default[sc.CAT],
                "train_time_s": t_mem, "inference_time_s": float(sum(t_inf.values())), "total_time_s": total})
            print(f"{setting} seed{seed}: memory bank {t_mem:.0f}s, total {total:.0f}s", flush=True)


if __name__ == "__main__":
    main()
