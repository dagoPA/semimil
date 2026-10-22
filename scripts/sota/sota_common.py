"""Shared helpers for the SOTA wrappers (imported inside each method's own venv).

Only numpy / torch / PIL are used so the module works in every third-party environment.
Every wrapper writes, per run, into ``<OUT_ROOT>/<method>/<setting>/seed<k>/``:
    image_scores.csv   name,label,score  (name = original MIIC file name, e.g. test_abnormal_00001.jpg)
    maps256.npy        float32 [N, 256, 256] anomaly maps, same row order as image_scores.csv
    run_info.json      config, train/inference time, number of training images
The metrics are computed afterwards, identically for every method, by ``evaluate.py``.
"""

from __future__ import annotations

import csv
import json
import os
import random
import time
from pathlib import Path

import numpy as np

TREE = Path("/autofs/space/crater_001/projects/micropath/semi/external_code/miic_mvtec")
CAT = "miic"
OUT_ROOT = Path("/autofs/space/crater_001/projects/micropath/semi/outputs/sota")
MAP = 256  # common map resolution for pixel metrics (as in scripts/localization.py)


def original_name(tree_path: str | Path) -> str:
    """Symlink name in the MVTec tree (.png) -> original MIIC file name (.jpg)."""
    return Path(tree_path).stem + ".jpg"


def train_paths() -> list[Path]:
    return sorted((TREE / CAT / "train" / "good").glob("*.png"))


SMOKE = int(os.environ.get("SMOKE", 0))  # >0: smoke test on the first SMOKE images of each class


def test_items() -> list[tuple[Path, int]]:
    """(tree path, label) for the full real test split: 1,272 normals then 116 defects."""
    good = sorted((TREE / CAT / "test" / "good").glob("*.png"))
    bad = sorted((TREE / CAT / "test" / "defect").glob("*.png"))
    if SMOKE:
        good, bad = good[:SMOKE], bad[:SMOKE]
    return [(p, 0) for p in good] + [(p, 1) for p in bad]


def subsample(paths: list[Path], n: int, seed: int) -> list[Path]:
    """Seeded random subsample (used only where the official budget cannot use all normals)."""
    rng = random.Random(seed)
    idx = sorted(rng.sample(range(len(paths)), n))
    return [paths[i] for i in idx]


def resize_maps(maps, size: int = MAP) -> np.ndarray:
    """[N,H,W] float maps -> [N,size,size] float32, bilinear (align_corners=False)."""
    import torch
    import torch.nn.functional as F
    out = []
    for i in range(0, len(maps), 64):
        x = torch.as_tensor(np.stack([np.asarray(m, dtype=np.float32) for m in maps[i:i + 64]]))[:, None]
        if x.shape[-1] != size or x.shape[-2] != size:
            x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
        out.append(x[:, 0].numpy().astype(np.float32))
    return np.concatenate(out, 0)


def run_dir(method: str, setting: str, seed: int) -> Path:
    if SMOKE:
        setting = "SMOKE_" + setting
    d = OUT_ROOT / method / setting / f"seed{seed}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_run(d: Path, names: list[str], labels: list[int], scores, maps256: np.ndarray, info: dict) -> None:
    assert len(names) == len(labels) == len(scores) == maps256.shape[0]
    assert maps256.shape[1:] == (MAP, MAP), maps256.shape
    expected = {original_name(p): y for p, y in test_items()}
    assert dict(zip(names, labels)) == expected, "test set mismatch (names/labels)"
    np.save(d / "maps256.npy", maps256.astype(np.float32))
    with (d / "image_scores.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "label", "score"])
        for n, y, s in zip(names, labels, scores):
            w.writerow([n, int(y), repr(float(s))])
    info = {**info, "saved": time.strftime("%Y-%m-%d %H:%M:%S"), "host": os.uname().nodename}
    (d / "run_info.json").write_text(json.dumps(info, indent=2, default=str))
    if os.environ.get("RUNDIR_LIST"):
        with open(os.environ["RUNDIR_LIST"], "a") as fh:
            fh.write(str(d) + "\n")
    print(f"saved {len(names)} test images -> {d}", flush=True)
