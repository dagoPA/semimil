"""Alarm-driven vs random acquisition of normals: learning curves and paired comparison.

Reads outputs/acquire_alarm/ and outputs/acquire_random/ (scripts/run_acquire.py):
results.json (per checkpoint metrics of the fused detector) and raw/<rep>_n<M>.npz
(test and calibration maps).  At every memory size present in both runs it recomputes the
fused score, reports AUROC, Sensitivity@(Specificity = 0.99) and small-defect sensitivity,
and a paired bootstrap 95 % CI of alarm minus random.  Reference lines: the exhaustive
bank of 23,688 images (outputs/bigbank_*/raw/n23688.npz) and the paper's 100k-token
baseline.  Writes outputs/acquire_results.json and outputs/report/assets/acquisition.{png,pdf}.

    uv run python scripts/analysis/acquire_eval.py
"""
import glob
import json
import re
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

R = Path('/autofs/space/crater_001/projects/micropath/semi')
RUNS = {'alarm': R / 'outputs/acquire_alarm', 'random': R / 'outputs/acquire_random_s0'}  # same initial memory
RANDOM_SEEDS = [R / f'outputs/acquire_random_s{k}' for k in range(3)]
REPS = ['L8', 'L10_pool3']
EXH = {'L8': R / 'outputs/bigbank_L8/raw/n23688.npz', 'L10_pool3': R / 'outputs/bigbank_L10_pool3/raw/n23688.npz'}
BASELINE = {'sens_spec99': 0.787, 'sens_small': 0.523, 'auroc': 0.975}  # 100k-token bank, 3 seeds (paper)

abn = sorted(p for p in glob.glob(str(R / 'MIIC/Anomaly_test/abnormal_img/*')) if p.lower().endswith(('.jpg', '.png')))
nrm = sorted(p for p in glob.glob(str(R / 'MIIC/Anomaly_test/normal_img/*')) if p.lower().endswith(('.jpg', '.png')))
nn = len(nrm)
areas = np.array([(np.asarray(Image.open(R / 'MIIC/Anomaly_test/abnormal_mask' / (Path(p).stem + '_mask.jpg'))
                              .convert('L')) > 127).mean() * 100 for p in abn])
y = np.r_[np.zeros(nn), np.ones(len(abn))]


def auroc(yy, s):
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    r = (np.cumsum(cnt) - (cnt - 1) / 2.0)[inv]
    n1 = (yy == 1).sum(); n0 = len(yy) - n1
    return (r[yy == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def metrics(s, idx=None):
    idx = np.arange(len(y)) if idx is None else idx
    yy, ss = y[idx], s[idx]
    thr = np.percentile(ss[yy == 0], 99)
    a = ss[yy == 1]; ar = areas[idx[yy == 1] - nn]
    return np.array([auroc(yy, ss), (a >= thr).mean(), (a[ar < 0.1] >= thr).mean()])


def fused(paths):
    zs = []
    for p in paths:
        z = np.load(p)
        t = z['test_map'].astype(np.float32).reshape(len(z['test_map']), -1).max(1)
        c = z['calib_map'].astype(np.float32).reshape(len(z['calib_map']), -1).max(1)
        zs.append((t - c.mean()) / c.std())
    return np.mean(zs, 0)


def sizes(run):
    return sorted({int(re.search(r'_n(\d+)\.npz', f).group(1)) for f in glob.glob(str(run / 'raw' / 'L8_n*.npz'))})


def boot_diff(a, b, n=2000):
    rng = np.random.RandomState(0); i0, i1 = np.arange(nn), np.arange(nn, len(y)); d = []
    for _ in range(n):
        idx = np.r_[rng.choice(i0, nn), rng.choice(i1, len(i1))]
        d.append(metrics(a, idx) - metrics(b, idx))
    return np.percentile(np.array(d), [2.5, 97.5], axis=0)


def main():
    ref = metrics(fused([EXH[r] for r in REPS]))
    curves = {}
    for name, run in RUNS.items():
        curves[name] = {m: metrics(fused([run / 'raw' / f'{r}_n{m}.npz' for r in REPS])) for m in sizes(run)}
    seeds = [{m: metrics(fused([run / 'raw' / f'{r}_n{m}.npz' for r in REPS])) for m in sizes(run)}
             for run in RANDOM_SEEDS if sizes(run)]
    common = sorted(set(curves['alarm']) & set(curves['random']))
    out = {'exhaustive': dict(zip(('auroc', 'sens_spec99', 'sens_small'), ref)), 'rows': []}
    print('%7s  %-26s  %-26s   95%% CI alarm-random: Sens99 / small' % ('images', 'alarm AUROC/Sens99/small', 'random AUROC/Sens99/small'))
    for m in common:
        a, b = curves['alarm'][m], curves['random'][m]
        lo, hi = boot_diff(fused([RUNS['alarm'] / 'raw' / f'{r}_n{m}.npz' for r in REPS]),
                           fused([RUNS['random'] / 'raw' / f'{r}_n{m}.npz' for r in REPS]))
        print('%7d  %.3f / %.3f / %.3f         %.3f / %.3f / %.3f          [%+.3f, %+.3f] / [%+.3f, %+.3f]'
              % (m, *a, *b, lo[1], hi[1], lo[2], hi[2]))
        out['rows'].append({'images': m, 'alarm': a.tolist(), 'random': b.tolist(),
                            'ci_diff_sens': [lo[1], hi[1]], 'ci_diff_small': [lo[2], hi[2]]})
    print('exhaustive 23,688: AUROC %.3f  Sens99 %.3f  small %.3f' % tuple(ref))
    json.dump(out, open(R / 'outputs/acquire_results.json', 'w'), indent=1)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, (k, lab, refk) in zip(axes, [(1, 'Sensitivity at specificity 0.99', 'sens_spec99'),
                                         (2, 'Small-defect sensitivity', 'sens_small'), (0, 'Image AUROC', 'auroc')]):
        xs = sorted(curves['alarm'])
        ax.plot(xs, [curves['alarm'][m][k] for m in xs], marker='o', ms=4, color='#c53030', label='alarm-driven')
        xr = sorted(set.intersection(*[set(c) for c in seeds])) if seeds else []
        if xr:
            v = np.array([[c[m][k] for m in xr] for c in seeds])
            ax.plot(xr, v.mean(0), marker='s', ms=4, color='#8c8c8c', label='random (%d orders)' % len(seeds))
            ax.fill_between(xr, v.min(0), v.max(0), color='#8c8c8c', alpha=0.2, lw=0)
        ax.axhline(ref[k], color='#2b6cb0', ls='--', lw=1.2, label='all 23,688 images')
        ax.axhline(BASELINE[refk], color='#444', ls=':', lw=1.2, label='100k-token bank')
        ax.set_xscale('log'); ax.set_xlabel('normal images in memory'); ax.set_title(lab, fontsize=10)
        ax.set_xticks([250, 500, 1000, 2000, 4000, 6000]); ax.set_xticklabels(['250', '500', '1k', '2k', '4k', '6k'])
        ax.minorticks_off()
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    out_png = R / 'outputs/report/assets/acquisition.png'
    fig.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(out_png.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    print('saved ->', out_png)


if __name__ == '__main__':
    main()
