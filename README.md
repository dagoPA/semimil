# semimil — Anomaly Detection in Integrated-Circuit Microscopy

**Unsupervised versus Weakly-Supervised Multiple-Instance Learning for
semiconductor (IC) inspection.**

Dagoberto Pulido Arias · Massachusetts General Hospital / Harvard Medical School ·
`dpulido@mgh.harvard.edu`

Reproducible code for the paper submitted to *IEEE Latin America Transactions*.
Every table and figure in the paper is produced by a YAML-driven script in
[`scripts/`](scripts) reading a config in [`configs/`](configs) — no command-line
arguments, for reproducibility.

---

## Abstract

Automated inspection of integrated circuits (IC) from microscopy images is a
highly imbalanced anomaly-detection problem: defects are rare, tiny, and
essentially unlabeled. We compare two method families under a Multiple-Instance
Learning (MIL) formulation, where each image is a *bag* of patches: (i) an
**unsupervised one-class detector** built on a memory bank of normal patches, and
(ii) a **weakly-supervised attention-MIL classifier** trained on synthetic
anomalies. On the MIIC dataset (25,276 images, 0.46 % anomalous) the feature
extractor dominates: self-supervised **DINOv2** features vastly outperform
supervised ImageNet ones with no extra training (image AUROC 0.923 → 0.972). A
cross-domain evaluation quantifies a domain gap (synthetic-trained models degrade
on real defects), and we report a **negative result**: self-supervised domain
adaptation does not help. Under a unified, leakage-free protocol against nine
methods — including FastFlow, STFPM and CFA reproduced with anomalib — the
one-class detector wins with statistical significance (DeLong, Holm, bootstrap).
We show that the previously reported state of the art on this dataset depends on
**selecting checkpoints using the test set**, and that the method **generalizes
unchanged to MVTec AD** (mean image AUROC 0.974 over 15 categories). Two
deployment questions are then settled empirically: **eight normal images** already
suffice to beat every baseline trained on all 23,888 of them, and **split
conformal** turns normal-only scores into a threshold that catches 68 % of real
defects at a guaranteed 1 % false-alarm rate, without a single label. By avoiding
labels, the approach supports deployable, low-waste inspection (UN SDGs 9 and 12).

## Key results

**Extractor dominates (one-class detector, no training):**

| Extractor | Image AUROC | Image AUPRC |
|---|---|---|
| WideResNet-50 (ImageNet) | 0.923 | 0.714 |
| **DINOv2 ViT-S/14** | **0.972** | **0.911** |

**Unified benchmark on the real MIIC test** (image AUROC, 95 % bootstrap CI,
DeLong+Holm vs. the best method; ‡ reproduced with anomalib):

| Method | AUROC [95 % CI] | p (Holm) | reported |
|---|---|---|---|
| **DINOv2 (one-class)** | **0.972 [.953–.988]** | ref | — |
| Attention-MIL | 0.958 [.937–.976] | 0.20 (n.s.) | — |
| WRN50 (one-class) | 0.923 [.899–.945] | <.001 | — |
| FastFlow ‡ | 0.908 [.873–.941] | <.001 | 0.976 |
| PatchCore ‡ | 0.904 [.865–.940] | <.001 | 0.933 |
| DINOv2-domain | 0.890 [.854–.922] | <.001 | — |
| PaDiM | 0.827 [.784–.868] | <.001 | 0.849 |
| STFPM ‡ | 0.797 [.750–.841] | <.001 | 0.963 |
| CFA ‡ | 0.755 [.703–.808] | <.001 | 0.990 |
| DRAEM | 0.593 [.531–.651] | <.001 | — |

**Why reproduced ≪ reported:** literature numbers use anomalib's default split,
which keeps half the test set (with real anomalies) as a validation set to save
the *best checkpoint* — i.e. model selection on the test set. Replicating only
that selection recovers the published numbers (STFPM 0.964 vs. 0.963; FastFlow
0.971 vs. 0.976), while our detector — which has no checkpoint to pick — leads
without ever touching the test set.

**Deployment budget — how many normal images does the bank need?**
(mean ± std over 5 seeds; no labels, no training)

| Normal images `k` | Image AUROC |
|---|---|
| 1 | 0.810 ± 0.112 |
| 8 | **0.941 ± 0.015** |
| 16 | 0.958 ± 0.021 |
| 128 | 0.965 ± 0.007 |
| 1000 | 0.975 ± 0.002 |

**8 normal images already beat every baseline trained on all 23,888** (FastFlow
0.908, PatchCore 0.904, PaDiM 0.827). Seed-to-seed std collapses from 0.112 at
k=1 to 0.015 at k=8, so below ~8 images the result is not just worse but
unpredictable.

**Contamination tolerance** — a pool of 1000 unlabeled images of which a fraction
ε is actually defective, with and without a label-free self-cleaning pass (drop
the most anomalous 10 %, rebuild):

| ε | synthetic, raw | synthetic, cleaned | real, raw | real, cleaned | contaminants caught |
|---|---|---|---|---|---|
| 0 % | 0.978 | 0.976 | 0.960 | 0.968 | — |
| 1 % | 0.975 | 0.972 | 0.951 | **0.963** | 0.72 |
| 5 % | 0.975 | 0.973 | 0.914 | 0.916 | 0.53 |
| 10 % | 0.972 | 0.970 | n/a | n/a | — |

Synthetic contaminants barely poison the bank (−0.006 AUROC at ε=10 %) while real
ones clearly do — independent evidence for the domain gap reported above. Self-
cleaning catches 72–80 % of *real* contaminants and costs nothing on a clean pool.
The real arm stops at ε=5 % because only 58 real anomalies are available as
contaminants (the other half stays in the test set, so contaminants never leak
into evaluation).

**Label-free operating point (split conformal).** Calibrating on 2,000 held-out
*normal* images only, the threshold `τ(α)` = the ⌈(n+1)(1−α)⌉-th smallest
calibration score carries a finite-sample, distribution-free guarantee
`P(score > τ) ≤ α`:

| Guaranteed false-alarm rate | Empirical | Real defects caught |
|---|---|---|
| 1 % | 0.0013 | **68 %** |
| 5 % | 0.0113 | **84 %** |
| 10 % | 0.0427 | 92 % |

Under exchangeability the guarantee is exact (α=0.05 → 0.0498). Deployed, it is
conservative because held-out train normals score higher than test normals
(Mann–Whitney p=3e-10), so the error falls on the safe side. On MVTec AD it
transfers on average but exposes two limits: `carpet`, `wood` and `toothbrush`
violate the nominal rate (train/test normals are not exchangeable there), and
α=0.01 is unreachable in 13 of 15 categories because calibration needs ≥ 1/α − 1
images.

**Generalization (one-class detector, DINOv2, no training):**

| Dataset | Image AUROC | Pixel AUROC | PRO |
|---|---|---|---|
| MIIC (semiconductors) | 0.972 | 0.999 | 0.981 |
| **MVTec AD (mean of 15)** | **0.974** | **0.970** | **0.917** |

## Repository layout

```
src/semimil/        installable package (config, data utils, seeding)
scripts/            one entry point per experiment (reads a config, no CLI args)
configs/            one YAML per script (paths, hyper-parameters)
pyproject.toml      uv/pip project; package `semimil`, Python >=3.11, ruff
```

## Install

```bash
uv sync                       # create the environment from uv.lock
# anomalib baselines run in an ephemeral overlay:
#   uv run --with anomalib --with certifi python scripts/run_anomalib.py
```

## Reproduce

Each result maps to one script + one config (`uv run python scripts/<x>.py`):

| Result | Script | Config |
|---|---|---|
| Extract MIIC | `extract_data.py` | `data.yaml` |
| Data/defect-size analysis | `analyze_size.py` | — |
| **One-class detector** (WRN50/DINOv2) | `run_patchcore.py` | `patchcore.yaml` |
| **Attention-MIL** | `train_mil.py` | `mil.yaml` |
| Cross-eval (real vs. synthetic) | `cross_eval.py` | `cross_eval.yaml` |
| Domain SSL (negative result) | `pretrain_ssl.py` | `ssl.yaml` |
| Multi-seed sweeps / ablations | `sweep.py` | `sweep.yaml` |
| Baseline PaDiM | `run_padim.py` | `padim.yaml` |
| Baseline DRAEM | `run_draem.py` | `draem.yaml` |
| Baselines FastFlow/STFPM/CFA | `run_anomalib.py` | `anomalib.yaml` |
| Baseline PatchCore | `run_patchcore_anomalib.py` | `anomalib.yaml` |
| Checkpoint-on-test diagnostic | `anomalib_epochsel.py` | `anomalib_epochsel.yaml` |
| **Deployment budget** (k-shot + contamination) | `deployment_budget.py` | `deployment_budget.yaml` |
| **Label-free conformal threshold** | `conformal_threshold.py` | `conformal.yaml` |
| DeLong test | `stats_tests.py` | — |
| Holm correction | `holm.py` | — |
| Bootstrap CIs | `bootstrap_ci.py` | — |
| Localization (pixel AUROC + PRO) | `localization.py`, `localization_anomalib.py` | — |
| **MVTec AD generalization** | `run_mvtec.py` | `mvtec.yaml` |
| Figures | `plot_sankey.py`, `plot_comparison.py`, `plot_ranking.py`, `plot_epochsel.py`, `plot_deployment_budget.py`, `plot_graphical_abstract.py` | — |

> **Paths.** Configs point to the shared project space
> (`/autofs/space/crater_001/projects/micropath/semi`) for data and outputs;
> edit the `data`/`output` fields to run elsewhere.

## Data

- **MIIC** (Huang *et al.*, ICIP 2021) — IC-microscopy anomaly detection.
- **MVTec AD** (Bergmann *et al.*, CVPR 2019) — external-validity benchmark.

Both are publicly available; see the paper for licensing and access.

## Citation

```bibtex
@article{pulido2025icmil,
  title   = {Anomaly Detection in Integrated-Circuit Microscopy:
             Unsupervised versus Weakly-Supervised Multiple-Instance Learning},
  author  = {Pulido Arias, Dagoberto},
  journal = {IEEE Latin America Transactions (under review)},
  year    = {2025}
}
```

## Notes

The manuscript source (LaTeX, IEEE template) is under `outputs/report/latex/` in
the project space. AI-based tools assisted with code and drafting; all
experiments, analyses and conclusions were verified by the author.
