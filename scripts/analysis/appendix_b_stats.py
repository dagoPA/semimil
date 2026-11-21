"""Numbers of Appendix B under the protocol of the main text.

Reads the one-class runs of scripts/run_appendix_oneclass.py (seeds 0-2, memory banks
drawn as for the main-text baseline), the attention-MIL run of scripts/train_mil.py and
the stored per-image scores of the classical baselines. The DINOv2 run reproduces the
baseline detector of the main text up to the floating-point nondeterminism of the GPU
(image AUROC 0.9779 against 0.9780 on the first seed, Pearson r 0.999997), so its image
scores and stitched maps are read from the main-text run (outputs/tiled/raw/g1_518_s*.npz)
and the numbers of both sections coincide; the reproduction supplies its synthetic-test
scores and the token maps of the top-k ablation. The script computes:
  - AUROC and AUPRC of every method (means over seeds for the one-class detectors), with
    95 % CIs from a stratified bootstrap of the first seed (2,000 resamples), and DeLong
    tests of each method against the DINOv2 one-class detector (first seed), with Holm
    correction (full comparison table and ranking figure);
  - AUROC, AUPRC, balanced accuracy and Cohen's kappa at the 99th percentile of the
    normal scores, on the real and the synthetic test (cross-evaluation table);
  - pixel AUROC and PRO of the one-class detectors, token maps stitched to 128 px as in
    scripts/run_tiled.py and upsampled to 256 px (localization table);
  - the ablations of the DINOv2 detector: memory size, top-k pooling, input resolution;
  - the sensitivity of the one-class detectors at the 99th percentile of the normal test
    scores, over every defect and over the defects below 0.1 % of the image, as in
    scripts/run_selfref.py (table of approaches that did not help).

Writes outputs/appendix_b/summary.json and, in the format that plot_ranking.py and
plot_epochsel.py read, outputs/appendix_b/stats/{bootstrap,delong}.json.

    uv run python scripts/analysis/appendix_b_stats.py
"""
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from holm import holm
from localization import pro_auc
from stats_tests import delong_test

R = Path("/autofs/space/crater_001/projects/micropath/semi")
OC = R / "outputs/appendix_oneclass/raw"
TILED = R / "outputs/tiled/raw"            # main-text baseline detector (scale g1_518)
MIL = R / "outputs/mil/mil_dinov2"
OUT = R / "outputs/appendix_b"
SEEDS, B, STITCH, MAP = (0, 1, 2), 2000, 128, 256
REF = "DINOv2 (one-class)"
ONECLASS = {REF: "dinov2", "WRN50 (one-class)": "wrn50", "DINOv2-dom (one-class)": "dinov2_domain"}
CSVS = {"Attention-MIL": MIL / "image_scores.csv",
        "PatchCore": R / "outputs/anomalib/patchcore/image_scores.csv",
        "FastFlow": R / "outputs/anomalib/fastflow/image_scores.csv",
        "STFPM": R / "outputs/anomalib/stfpm/image_scores.csv",
        "CFA": R / "outputs/anomalib/cfa/image_scores.csv",
        "PaDiM": R / "outputs/padim/image_scores.csv",
        "DRAEM": R / "outputs/draem/image_scores.csv"}

ext = {".jpg", ".jpeg", ".png"}
norm = sorted(p for p in (R / "MIIC/Anomaly_test/normal_img").iterdir() if p.suffix.lower() in ext)
abn = sorted(p for p in (R / "MIIC/Anomaly_test/abnormal_img").iterdir() if p.suffix.lower() in ext)
names = [p.name for p in norm + abn]
y = np.r_[np.zeros(len(norm), int), np.ones(len(abn), int)]
syn_y = np.r_[np.zeros(135, int), np.ones(135, int)]
area = np.array([(np.asarray(Image.open(R / f"MIIC/Anomaly_test/abnormal_mask/{p.stem}_mask.jpg")
                             .convert("L")) > 127).mean() * 100 for p in abn])


def runs(name):
    """Per-seed arrays of a one-class run; DINOv2 takes the main-text image scores and maps."""
    out = []
    for s in SEEDS:
        z = dict(np.load(OC / f"{name}_s{s}.npz"))
        if name == "dinov2":
            t = np.load(TILED / f"g1_518_s{s}.npz")
            z["test_score"], z["stitched"] = t["test_score"], t["test_map"]
        out.append(z)
    return out


def stitched(r):
    """Image maps at 128 px, as run_tiled.stitch builds them from the token maps of one tile."""
    if "stitched" in r:
        return r["stitched"].astype(np.float32)
    t = torch.from_numpy(r["test_map"].astype(np.float32))[:, None]
    up = F.interpolate(t, size=(STITCH, STITCH), mode="bilinear", align_corners=False)
    return up[:, 0].numpy()


def read_csv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def csv_scores(path):
    d = {r["name"]: float(r["score"]) for r in read_csv(path)}
    return np.array([d[n] for n in names])


def boot_ci(s, seed=0):
    rng = np.random.RandomState(seed)
    i0, i1 = np.where(y == 0)[0], np.where(y == 1)[0]
    v = []
    for _ in range(B):
        idx = np.r_[rng.choice(i0, len(i0)), rng.choice(i1, len(i1))]
        v.append([roc_auc_score(y[idx], s[idx]), average_precision_score(y[idx], s[idx])])
    lo, hi = np.percentile(v, 2.5, 0), np.percentile(v, 97.5, 0)
    return [float(lo[0]), float(hi[0])], [float(lo[1]), float(hi[1])]


def op_metrics(yy, s, pct=99):
    """AUROC, AUPRC, balanced accuracy and kappa; threshold at the pct-th normal percentile."""
    pred = (s >= np.percentile(s[yy == 0], pct)).astype(int)
    return np.array([roc_auc_score(yy, s), average_precision_score(yy, s),
                     balanced_accuracy_score(yy, pred), cohen_kappa_score(yy, pred)])


def sensitivity(s):
    """Sensitivity over all defects and over those below 0.1 % of the image."""
    hit = s[y == 1] >= np.percentile(s[y == 0], 99)
    return [float(hit.mean()), float(hit[area < 0.1].mean())]


def localization(run, i):
    """Pixel AUROC and PRO of seed i of a one-class run, maps upsampled to 256 px."""
    masks = np.zeros((len(names), MAP, MAP), np.uint8)
    for k, p in enumerate(abn):
        m = Image.open(R / f"MIIC/Anomaly_test/abnormal_mask/{p.stem}_mask.jpg").convert("L")
        masks[len(norm) + k] = np.asarray(m.resize((MAP, MAP), Image.NEAREST)) > 127
    t = torch.from_numpy(stitched(runs(run)[i]))[:, None]
    up = F.interpolate(t, size=(MAP, MAP), mode="bilinear", align_corners=False)[:, 0].numpy()
    return [roc_auc_score(masks.ravel(), up.ravel()), pro_auc(up, masks)]


def topk_score(maps, k):
    flat = maps.reshape(len(maps), -1).astype(np.float32)
    return np.sort(flat, axis=1)[:, -k:].mean(1)


def mean_std(vals):
    v = np.array(vals)
    return [float(x) for x in v.mean(0)], [float(x) for x in v.std(0)]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "stats").mkdir(exist_ok=True)
    summary = {}

    # ---- full comparison: AUROC/AUPRC, CIs of the first seed, DeLong against the reference
    first, table = {}, {}
    for label, run in ONECLASS.items():
        rs = runs(run)
        m, sd = mean_std([[roc_auc_score(y, r["test_score"]),
                           average_precision_score(y, r["test_score"])] for r in rs])
        first[label] = rs[0]["test_score"].astype(np.float64)
        table[label] = {"auroc": m[0], "auprc": m[1], "auroc_std": sd[0], "auprc_std": sd[1],
                        "n_seeds": len(rs)}
    for label, path in CSVS.items():
        s = csv_scores(path)
        first[label] = s
        table[label] = {"auroc": float(roc_auc_score(y, s)),
                        "auprc": float(average_precision_score(y, s)), "n_seeds": 1}
    for label, s in first.items():
        table[label]["auroc_ci"], table[label]["auprc_ci"] = boot_ci(s)
    comps = []
    for label, s in first.items():
        if label == REF:
            continue
        a_ref, a, z, p = delong_test(y, first[REF], s)
        comps.append({"method": label, "auroc": table[label]["auroc"], "delta_vs_ref": a_ref - a,
                      "p_delong": p, "z": z})
    for c, ph in zip(comps, holm([c["p_delong"] for c in comps]), strict=True):
        c["p_holm"] = ph
    (OUT / "stats/bootstrap.json").write_text(json.dumps({"n_boot": B, "results": [
        {"method": k, **v} for k, v in table.items()]}, indent=2))
    (OUT / "stats/delong.json").write_text(json.dumps({
        "reference": REF, "ref_auroc": table[REF]["auroc"], "comparisons": comps}, indent=2))
    summary["comparison"] = {"table": table, "delong": comps}

    # ---- cross-evaluation: real and synthetic test, at the operating point
    cross = {}
    for label, run in (("One-class (WRN50)", "wrn50"), ("One-class (DINOv2)", "dinov2")):
        rs = runs(run)
        cross[label] = {"real": mean_std([op_metrics(y, r["test_score"]) for r in rs])[0]}
        if "syn_score" in rs[0]:
            cross[label]["synthetic"] = mean_std([op_metrics(syn_y, r["syn_score"]) for r in rs])[0]
    syn = read_csv(MIL / "synthetic_scores.csv")
    sy = np.array([int(r["label"]) for r in syn])
    ss = np.array([float(r["score"]) for r in syn])
    cross["Attention-MIL"] = {"real": [float(x) for x in op_metrics(y, first["Attention-MIL"])],
                              "synthetic": [float(x) for x in op_metrics(sy, ss)]}
    summary["cross"] = {"columns": ["auroc", "auprc", "balanced_accuracy", "kappa"], "rows": cross}

    # ---- localization of the one-class detectors, one process per run and seed
    with ProcessPoolExecutor(len(ONECLASS) * len(SEEDS)) as pool:
        jobs = {label: [pool.submit(localization, run, i) for i in range(len(SEEDS))]
                for label, run in ONECLASS.items()}
        loc = {label: mean_std([j.result() for j in js])[0] for label, js in jobs.items()}
    summary["localization"] = {"columns": ["pixel_auroc", "pro"], "rows": loc}

    # ---- ablations of the DINOv2 detector
    def auroc_auprc(scores):
        return mean_std([[roc_auc_score(y, s), average_precision_score(y, s)] for s in scores])
    base = runs("dinov2")
    abl = {"memory_images": {}, "top_k": {}, "resolution": {}}
    for n in (200, 500, 1000, 2000):
        run = "dinov2" if n == 1000 else f"dinov2_n{n}"
        abl["memory_images"][n] = auroc_auprc([r["test_score"] for r in runs(run)])
    for k in (1, 3, 5, 10):  # k = 1 is the image score itself
        abl["top_k"][k] = auroc_auprc([r["test_score"] if k == 1 else topk_score(r["test_map"], k)
                                       for r in base])
    for res, run in ((224, "dinov2_r224"), (378, "dinov2_r378"), (518, "dinov2")):
        abl["resolution"][res] = auroc_auprc([r["test_score"] for r in runs(run)])
    summary["ablations"] = abl

    # ---- sensitivity at specificity 0.99 of the one-class detectors
    sens = {label: mean_std([sensitivity(r["test_score"]) for r in runs(run)])[0]
            for label, run in ONECLASS.items()}
    summary["sensitivity"] = {"columns": ["all", "small"], "rows": sens}

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    print("== full comparison (mean over seeds; CI and DeLong on the first seed)")
    for label, v in sorted(table.items(), key=lambda kv: -kv[1]["auroc"]):
        c = next((c for c in comps if c["method"] == label), None)
        lo, hi = v["auroc_ci"]
        plo, phi = v["auprc_ci"]
        print(f"  {label:20s} AUROC {v['auroc']:.3f} [{lo:.3f}, {hi:.3f}]  "
              f"AUPRC {v['auprc']:.3f} [{plo:.3f}, {phi:.3f}]  "
              + (f"p_holm {c['p_holm']:.2g}" if c else "ref"))
    print("== cross-evaluation (AUROC, AUPRC, BA, kappa)")
    for label, v in cross.items():
        for dom, vals in v.items():
            print(f"  {label:20s} {dom:9s} " + " ".join(f"{x:.3f}" for x in vals))
    print("== localization (pixel AUROC, PRO)")
    for label, v in loc.items():
        print(f"  {label:20s} {v[0]:.4f} {v[1]:.3f}")
    print("== sensitivity at specificity 0.99 (all, small)")
    for label, v in sens.items():
        print(f"  {label:20s} {v[0]:.3f} {v[1]:.3f}")
    print("== ablations (AUROC mean+-std, AUPRC mean+-std)")
    for kind, d in abl.items():
        for key, (m, sd) in d.items():
            print(f"  {kind:14s} {key!s:5s} {m[0]:.3f}+-{sd[0]:.3f}  {m[1]:.3f}+-{sd[1]:.3f}")


if __name__ == "__main__":
    main()
