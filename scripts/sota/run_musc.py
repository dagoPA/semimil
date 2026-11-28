#!/usr/bin/env python
"""MuSc (ICLR 2024) on MIIC: zero-shot mutual scoring of the unlabeled test images.

Runs the official pipeline of ``models/musc.py::MuSc.make_category_data`` with the README
defaults: CLIP ViT-L/14-336 (OpenAI weights) at 518 px, patch tokens of layers 6, 12, 18 and
24, local neighborhood aggregation (LNAMD) with r = 1, 3 and 5, mutual scoring (MSM) of every
test image against all the others with the interval average over the lowest 30 % of the
image-to-image distances, and the RsCIN re-scoring of the image scores with k = 1, 2, 3 (the
MVTec AD setting).  The 1,388 test images are scored as one batch (divide_num = 1), so no
training image is used and the result does not depend on a seed.  The loop is copied from
``make_category_data`` without its metric and visualization code, which would hold the full
518x518 maps of every image in memory. Each layer's features are released once the layer is
scored, so that, with expandable CUDA segments, the whole test set fits on one 40 GB GPU.
The image score is the maximum of the 37x37 map, equal to the maximum of the map upsampled
with align_corners=True as in the official code.
Pixel maps: the 37x37 map upsampled bilinearly (align_corners=True) to 256x256.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

REPO = Path("/autofs/space/crater_001/projects/micropath/semi/external_code/musc")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sota_common as sc  # noqa: E402

sys.path[:0] = [str(REPO), str(REPO / "models" / "backbone")]
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from models.modules._LNAMD import LNAMD  # noqa: E402
from models.modules._MSM import MSM  # noqa: E402
from models.modules._RsCIN import RsCIN  # noqa: E402
from models.musc import MuSc  # noqa: E402

CFG = {
    "datasets": {"dataset_name": "mvtec_ad", "data_path": str(sc.TREE), "class_name": sc.CAT,
                 "img_resize": 518, "divide_num": 1},
    "models": {"backbone_name": "ViT-L-14-336", "pretrained": "openai", "batch_size": 8,
               "feature_layers": [5, 11, 17, 23], "r_list": [1, 3, 5]},
    "device": 0,
    "testing": {"output_dir": "", "vis": False, "vis_type": "single_norm", "save_excel": False},
}
K_SCORE = [1, 2, 3]  # RsCIN neighbors for MVTec AD and other datasets in the official code


@torch.no_grad()
def score_batch(model: MuSc):
    """Official feature extraction, LNAMD and MSM over the whole test set (divide_num = 1)."""
    ds = model.load_datasets(sc.CAT, divide_num=1, divide_iter=0)
    if sc.SMOKE:
        keep = [t for t in ds.data_to_iterate if t[1] == "good"][: sc.SMOKE]
        ds.data_to_iterate = keep + [t for t in ds.data_to_iterate if t[1] != "good"][: sc.SMOKE]
    loader = torch.utils.data.DataLoader(ds, batch_size=model.batch_size, shuffle=False,
                                         num_workers=0, pin_memory=True)
    paths, class_tokens, tokens = [], [], []
    t0 = time.time()
    for info in loader:
        paths.extend(info["image_path"])
        with torch.cuda.amp.autocast():
            x = info["image"].to(torch.float).to(model.device)
            image_features, patch_tokens = model.clip_model.encode_image(x, model.features_list)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            patch_tokens = [patch_tokens[l].cpu() for l in range(len(model.features_list))]
        class_tokens.extend(image_features[b].squeeze().cpu().numpy() for b in range(len(x)))
        tokens.append(patch_tokens)
    t_extract = time.time() - t0
    print(f"extracted {len(paths)} images in {t_extract:.0f}s", flush=True)

    dim = tokens[0][0].shape[-1]
    maps_r = []
    for r in model.r_list:
        lnamd = LNAMD(device=model.device, r=r, feature_dim=dim, feature_layer=model.features_list)
        z_layers = {}
        for pt in tokens:
            with torch.cuda.amp.autocast():
                f = lnamd._embed([p.to(model.device) for p in pt])
                f /= f.norm(dim=-1, keepdim=True)
                for l in range(len(model.features_list)):
                    z_layers.setdefault(l, []).append(f[:, :, l, :].contiguous())
        maps_l = []
        for l in list(z_layers):
            z = torch.cat(z_layers.pop(l), dim=0).to(model.device)  # frees this layer's list
            maps_l.append(MSM(Z=z, device=model.device, topmin_min=0, topmin_max=0.3).cpu())
            del z
            torch.cuda.empty_cache()
        maps_r.append(torch.stack(maps_l).mean(0))
        del z_layers
        torch.cuda.empty_cache()
        print(f"r={r} done ({time.time() - t0:.0f}s)", flush=True)
    maps = torch.stack(maps_r).mean(0)  # [N, L] patch scores
    return paths, np.array(class_tokens), maps, t_extract, time.time() - t0


def main() -> None:
    d = sc.run_dir("musc", "zeroshot_vitl14_336_r518", 0)
    CFG["testing"]["output_dir"] = str(d / "raw")
    t0 = time.time()
    model = MuSc(CFG, seed=0)
    paths, cls, maps, t_extract, t_score = score_batch(model)
    n, L = maps.shape
    g = int(round(L ** 0.5))
    grid = maps.view(n, 1, g, g).float()
    ac_score = grid.reshape(n, -1).max(1).values.double().numpy()
    scores = RsCIN(ac_score, list(cls), k_list=K_SCORE)
    up = F.interpolate(grid, size=(sc.MAP, sc.MAP), mode="bilinear", align_corners=True)[:, 0].numpy()
    by_name = {sc.original_name(p): i for i, p in enumerate(paths)}
    names, labels, sc_list, maps256 = [], [], [], []
    for p, y in sc.test_items():
        i = by_name[sc.original_name(p)]
        names.append(sc.original_name(p)); labels.append(y)
        sc_list.append(float(scores[i])); maps256.append(up[i])
    sc.save_run(d, names, labels, sc_list, np.stack(maps256).astype(np.float32), {
        "method": "MuSc", "repo_commit": "b76b93d", "setting": "zero-shot, whole test set as one batch",
        "seed": 0, "n_train_images": 0, "model": "ViT-L-14-336 (openai)", "resolution": 518,
        "feature_layers": CFG["models"]["feature_layers"], "r_list": CFG["models"]["r_list"],
        "msm_interval": [0, 0.3], "rscin_k": K_SCORE, "divide_num": 1,
        "extract_time_s": t_extract, "inference_time_s": t_score, "total_time_s": time.time() - t0})


if __name__ == "__main__":
    main()
