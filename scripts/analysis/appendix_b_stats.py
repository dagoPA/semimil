"""Numbers of Appendix B, and of the token-size and pooling rows of Appendix A.

Every one-class detector follows the protocol of the main-text baseline detector. The
runs of scripts/run_appendix_oneclass.py draw their memory banks as run_tiled.py does
(seeds 0-2, outside the calibration images). The DINOv2 run reproduces the baseline
detector up to the floating-point nondeterminism of the GPU (image AUROC 0.9779 against
0.9780 on the first seed, Pearson r 0.999997), so its image scores are read from the
main-text run (outputs/tiled/raw/g1_518_s*.npz) and the numbers of both sections
coincide; the reproduction supplies its synthetic-test scores and the token maps of the
pooling rows. The proposed detector is the fused z-score of the two exhaustive runs
(outputs/bigbank_L8, outputs/bigbank_L10_pool3), as in plot_headline.py. The script
computes:
  - AUROC and AUPRC of every method (means over seeds for the baseline detectors), with
    95 % CIs from a stratified bootstrap of the first seed (2,000 resamples), and DeLong
    tests of each method against the proposed detector, with Holm correction (full
    comparison table and ranking figure);
  - AUROC, AUPRC, sensitivity, balanced accuracy and Cohen's kappa at the 99th percentile
    of the normal scores, on the real and the synthetic test (real vs. synthetic table);
  - for the baseline detectors and for the token-size and pooling variants of the DINOv2
    one, AUROC, AUPRC and the sensitivity at the 99th percentile of the normal test
    scores over every defect and over the defects below 0.1 % of the image (table of
    approaches that did not help).

Writes outputs/appendix_b/summary.json and, in the format that plot_ranking.py and
plot_epochsel.py read, outputs/appendix_b/stats/{bootstrap,delong}.json.

    NUMPY_MADVISE_HUGEPAGE=0 uv run python scripts/analysis/appendix_b_stats.py
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from holm import holm
from stats_tests import delong_test

R = Path("/autofs/space/crater_001/projects/micropath/semi")
OC = R / "outputs/appendix_oneclass/raw"
TILED = R / "outputs/tiled/raw"            # main-text baseline detector (scale g1_518)
MIL = R / "outputs/mil/mil_dinov2"
OUT = R / "outputs/appendix_b"
SEEDS, B = (0, 1, 2), 2000
REF = "Proposed"
PROPOSED_RUNS = ("bigbank_L8", "bigbank_L10_pool3")
BASELINES = {"Baseline (DINOv2)": "dinov2", "Baseline (WRN50)": "wrn50",
             "Baseline (adapted DINOv2)": "dinov2_domain"}
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
area = np.array([(np.asarray(Image.open(R / f"MIIC/Anomaly_test/abnormal_mask/{p.stem}_mask.jpg")
                             .convert("L")) > 127).mean() * 100 for p in abn])


def runs(name):
    """Per-seed arrays of a one-class run; DINOv2 takes the main-text image scores."""
    out = []
    for s in SEEDS:
        z = dict(np.load(OC / f"{name}_s{s}.npz"))
        if name == "dinov2":
            z["test_score"] = np.load(TILED / f"g1_518_s{s}.npz")["test_score"]
        out.append(z)
    return out


def proposed_scores():
    """Fused z-score of the proposed detector, standardized on the calibration images."""
    zs = []
    for run in PROPOSED_RUNS:
        z = np.load(R / f"outputs/{run}/raw/n23688.npz")
        t = z["test_map"].astype(np.float32).reshape(len(z["test_map"]), -1).max(1)
        c = z["calib_map"].astype(np.float32).reshape(len(z["calib_map"]), -1).max(1)
        zs.append((t - c.mean()) / c.std())
    return np.mean(zs, 0).astype(np.float64)


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
    """AUROC, AUPRC, sensitivity, balanced accuracy and kappa at the pct-th normal percentile."""
    pred = (s >= np.percentile(s[yy == 0], pct)).astype(int)
    return np.array([roc_auc_score(yy, s), average_precision_score(yy, s), pred[yy == 1].mean(),
                     balanced_accuracy_score(yy, pred), cohen_kappa_score(yy, pred)])


def operating_point(s):
    """AUROC, AUPRC and sensitivity over all defects and over those below 0.1 % of the image."""
    hit = s[y == 1] >= np.percentile(s[y == 0], 99)
    return [roc_auc_score(y, s), average_precision_score(y, s), float(hit.mean()),
            float(hit[area < 0.1].mean())]


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
    first = {REF: proposed_scores()}
    table = {REF: {"auroc": float(roc_auc_score(y, first[REF])),
                   "auprc": float(average_precision_score(y, first[REF])), "n_seeds": 1}}
    for label, run in BASELINES.items():
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

    # ---- real vs. synthetic test, at the operating point
    cols = ["auroc", "auprc", "sensitivity", "balanced_accuracy", "kappa"]
    cross = {}
    for label in ("Baseline (WRN50)", "Baseline (DINOv2)"):
        rs = runs(BASELINES[label])
        cross[label] = {"real": mean_std([op_metrics(y, r["test_score"]) for r in rs])[0]}
        if "syn_score" in rs[0]:
            half = len(rs[0]["syn_score"]) // 2  # originals, then anomalies
            syn_y = np.r_[np.zeros(half), np.ones(half)]
            cross[label]["synthetic"] = mean_std([op_metrics(syn_y, r["syn_score"]) for r in rs])[0]
    syn = read_csv(MIL / "synthetic_scores.csv")
    sy = np.array([int(r["label"]) for r in syn])
    ss = np.array([float(r["score"]) for r in syn])
    cross["Attention-MIL"] = {"real": [float(x) for x in op_metrics(y, first["Attention-MIL"])],
                              "synthetic": [float(x) for x in op_metrics(sy, ss)]}
    summary["cross"] = {"columns": cols, "rows": cross}

    # ---- operating point of the baseline detectors and of the token-size and pooling variants
    variants = {label: [operating_point(r["test_score"]) for r in runs(run)]
                for label, run in BASELINES.items()}
    for res in (224, 378):
        variants[f"Input at {res} px"] = [operating_point(r["test_score"])
                                         for r in runs(f"dinov2_r{res}")]
    base = runs("dinov2")
    for k in (3, 5, 10):
        variants[f"Mean of the top {k} patches"] = [operating_point(topk_score(r["test_map"], k))
                                                    for r in base]
    op = {label: mean_std(v) for label, v in variants.items()}
    summary["operating_point"] = {"columns": ["auroc", "auprc", "sens", "sens_small"], "rows": op}

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    print("== full comparison (mean over seeds; CI and DeLong on the first seed)")
    for label, v in sorted(table.items(), key=lambda kv: -kv[1]["auroc"]):
        c = next((c for c in comps if c["method"] == label), None)
        lo, hi = v["auroc_ci"]
        plo, phi = v["auprc_ci"]
        print(f"  {label:26s} AUROC {v['auroc']:.4f} [{lo:.3f}, {hi:.3f}]  "
              f"AUPRC {v['auprc']:.4f} [{plo:.3f}, {phi:.3f}]  "
              + (f"p {c['p_delong']:.2g} p_holm {c['p_holm']:.2g}" if c else "ref"))
    print("== real vs. synthetic (" + ", ".join(cols) + ")")
    for label, v in cross.items():
        for dom, vals in v.items():
            print(f"  {label:20s} {dom:9s} " + " ".join(f"{x:.4f}" for x in vals))
    print("== operating point (AUROC, AUPRC, sens, sens small; mean +- std over seeds)")
    for label, (m, sd) in op.items():
        vals = "  ".join(f"{a:.4f}+-{b:.3f}" for a, b in zip(m, sd, strict=True))
        print(f"  {label:28s} {vals}")


if __name__ == "__main__":
    main()
