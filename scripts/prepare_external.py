#!/usr/bin/env python
"""Build leakage-free manifests for the two external semiconductor datasets.

Carinthia-S (Kofler & Hasic, Zenodo 10.5281/zenodo.16895427, CC BY 4.0)
    480x480 production SEM images of unpatterned wafers, six classes; class 6 is
    "no visible defect" (227 images, empty masks), classes 1-5 are defects (4,364).
    Only 227 normals exist, so they are split into K folds: each fold's memory bank
    uses the other folds' normals and is tested on its own held-out normals plus
    every defect image.  Each normal is therefore scored by a bank that never saw it.

Texture-AD Wafer (Lei et al., arXiv:2409.06367, Apache-2.0)
    512x512 crops of optical photographs of patterned wafers, 14 subclasses, MVTec
    layout.  The released split is not usable as is: almost every test crop comes
    from a photograph that also produced training crops, and most overlap a training
    crop in pixels.  The filename carries the source photograph (Image_<timestamp>)
    and the crop offset, so each subclass is re-split by photograph into K folds
    (cross-fitting): for fold k, every crop of the fold-k photographs (normal and
    defective, from the released train and test folders) is tested against a memory
    bank built from the normal crops of the other folds' photographs.  Almost every
    photograph contains defects, so this uses every defective crop exactly once.

Writes ``<output.dir>/carinthia.csv`` and ``<output.dir>/texturead.csv`` plus a JSON
summary.  Argument-free: reads ``configs/external.yaml`` (or ``SEMIMIL_CONFIG``).
"""

from __future__ import annotations

import csv
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(os.environ.get("SEMIMIL_CONFIG", REPO_ROOT / "configs" / "external.yaml"))
TAD_NAME = re.compile(r"^(Image_\d+)(?:_gray|_denoised)?_sx(\d+)_sy(\d+)_ps(\d+)\.png$")


def carinthia(cfg: dict) -> list[dict]:
    root = Path(cfg["root"])
    rows = list(csv.DictReader((root / "carinthia-s.csv").open(), delimiter=";"))
    normals = sorted(r["filename"] for r in rows if r["label"] == str(cfg["normal_label"]))
    rng = random.Random(cfg["seed"])
    rng.shuffle(normals)
    fold_of = {f: i % cfg["n_folds"] for i, f in enumerate(normals)}
    out = []
    for r in rows:
        normal = r["label"] == str(cfg["normal_label"])
        out.append({
            "image": str(root / r["image_path"]), "mask": str(root / r["mask_path"]),
            "label": 0 if normal else 1, "cls": r["label"],
            "fold": fold_of[r["filename"]] if normal else -1,
        })
    return out


def texturead(cfg: dict) -> list[dict]:
    root = Path(cfg["root"])
    crops = defaultdict(list)  # subclass -> list of crop records
    for split in ("train", "test"):
        for p in sorted(root.glob(f"*/{split}/*/*.png")):
            sub, kind = p.parts[-4], p.parts[-2]
            m = TAD_NAME.match(p.name)
            if not m:
                raise ValueError(f"unexpected Texture-AD filename: {p}")
            mask = root / sub / "ground_truth" / kind / p.name if kind != "good" else None
            crops[sub].append({"image": str(p), "mask": str(mask) if mask else "",
                               "label": int(kind != "good"), "cls": sub, "photo": m.group(1),
                               "released_split": split})
    rng = random.Random(cfg["seed"])
    out = []
    for sub in sorted(crops, key=int):
        photos = sorted({c["photo"] for c in crops[sub]})
        rng.shuffle(photos)
        fold_of = {ph: i % cfg["n_folds"] for i, ph in enumerate(photos)}
        out.extend({**c, "fold": fold_of[c["photo"]]} for c in crops[sub])
    return out


def mvtec(cfg: dict) -> list[dict]:
    """MVTec AD: one task per category; train/good is the normal pool, test as released."""
    root = Path(cfg["root"])
    out = []
    for cat in sorted(d.name for d in root.iterdir() if (d / "train" / "good").is_dir()):
        for p in sorted((root / cat / "train" / "good").glob("*.png")):
            out.append({"image": str(p), "mask": "", "label": 0, "cls": cat, "split": "train"})
        for d in sorted((root / cat / "test").iterdir()):
            for p in sorted(d.glob("*.png")):
                bad = d.name != "good"
                mask = root / cat / "ground_truth" / d.name / f"{p.stem}_mask.png" if bad else ""
                out.append({"image": str(p), "mask": str(mask), "label": int(bad), "cls": cat,
                            "split": "test"})
    return out


def write(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    cfg = yaml.safe_load(CONFIG_PATH.open())
    out = Path(cfg["output"]["dir"])
    out.mkdir(parents=True, exist_ok=True)
    summary = {}

    car = carinthia(cfg["carinthia"])
    write(car, out / "carinthia.csv")
    summary["carinthia"] = {
        "images": len(car), "per_class": dict(Counter(r["cls"] for r in car)),
        "normals_per_fold": dict(Counter(r["fold"] for r in car if r["label"] == 0)),
    }

    tad = texturead(cfg["texturead"])
    write(tad, out / "texturead.csv")
    per = {}
    for sub in sorted({r["cls"] for r in tad}, key=int):
        rs = [r for r in tad if r["cls"] == sub]
        per[sub] = {}
        for k in range(cfg["texturead"]["n_folds"]):
            te = [r for r in rs if r["fold"] == k]
            tr = [r for r in rs if r["fold"] != k and r["label"] == 0]
            assert not {r["photo"] for r in te} & {r["photo"] for r in tr}, f"leak in {sub}"
            per[sub][f"fold{k}"] = {
                "bank_pool_normal": len(tr), "test_normal": sum(r["label"] == 0 for r in te),
                "test_defect": sum(r["label"] == 1 for r in te),
                "test_photos": len({r["photo"] for r in te}),
            }
    summary["texturead"] = per

    if "mvtec" in cfg:
        mv = mvtec(cfg["mvtec"])
        write(mv, out / "mvtec.csv")
        summary["mvtec"] = {c: {"train": sum(r["cls"] == c and r["split"] == "train" for r in mv),
                                "test_normal": sum(r["cls"] == c and r["split"] == "test" and r["label"] == 0 for r in mv),
                                "test_defect": sum(r["cls"] == c and r["label"] == 1 for r in mv)}
                            for c in sorted({r["cls"] for r in mv})}
    (out / "manifest_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
