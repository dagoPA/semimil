"""Evaluate structure-conditional (Mondrian) calibration of exhaustive-bank patch distances.

For each representation and number of structure types K (labels from run_structcal.py),
patch distances d are re-scored within their type c using the calibration-normal
patches of that type (never in the bank):
    z      (d - mean_c) / std_c
    robust (d - median_c) / IQR_c
    conf   -log10 p,  p = (1 + #{calib_c >= d}) / (n_c + 1), with an exponential tail
           beyond the calibration maximum so that p is not floored at 1/(n_c+1)
Image score = max over patches. Fusion of two representations = mean of their patch
scores before the max (both on comparable, type-calibrated scales).
Reports AUROC, AUPRC, recall at the 99th percentile of normal test scores (all / small /
medium), and a nested 2-fold estimate over the choice of (variant, K).

numpy + PIL only:  python3 -I scripts/analysis/structcal_eval.py
"""
import glob
import itertools
import json

import numpy as np
from PIL import Image

R = '/autofs/space/crater_001/projects/micropath/semi/'
REPS = {'L8': 'bigbank_L8', 'L10_pool3': 'bigbank_L10_pool3', 'L12': 'bigbank'}
KS = [4, 8, 16, 32, 64]


def auroc(y, s):
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    r = (np.cumsum(cnt) - (cnt - 1) / 2.0)[inv]
    n1 = (y == 1).sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def ap(y, s):
    o = np.argsort(-s, kind='mergesort'); yy = y[o]; tp = np.cumsum(yy)
    return (tp / np.arange(1, len(yy) + 1) * yy).sum() / yy.sum()


abn = sorted(p for p in glob.glob(R + 'MIIC/Anomaly_test/abnormal_img/*') if p.lower().endswith(('.jpg', '.png')))
nrm = sorted(p for p in glob.glob(R + 'MIIC/Anomaly_test/normal_img/*') if p.lower().endswith(('.jpg', '.png')))
nn = len(nrm)
areas = np.array([(np.asarray(Image.open(R + 'MIIC/Anomaly_test/abnormal_mask/%s_mask.jpg'
                                         % p.split('/')[-1].rsplit('.', 1)[0]).convert('L')) > 127).mean() * 100 for p in abn])
y = np.r_[np.zeros(nn), np.ones(len(abn))]


def metrics(score, idx=None):
    idx = np.arange(len(y)) if idx is None else idx
    yy, s = y[idx], score[idx]
    n, a = s[yy == 0], s[yy == 1]; thr = np.percentile(n, 99); ar = areas[idx[yy == 1] - nn]
    return np.array([auroc(yy, s), ap(yy, s), (a >= thr).mean(), (a[ar < .1] >= thr).mean(),
                     (a[(ar >= .1) & (ar < 1)] >= thr).mean()])


def patch_scores(rep, k, variant):
    z = np.load(R + 'outputs/%s/raw/n23688.npz' % REPS[rep])
    td = z['test_map'].astype(np.float64).reshape(len(z['test_map']), -1)
    cd = z['calib_map'].astype(np.float64).reshape(len(z['calib_map']), -1)
    if variant == 'raw':
        return td
    lab = np.load(R + 'outputs/structcal/%s/labels_k%d.npz' % (rep, k))
    tl, cl = lab['test'].astype(int), lab['calib'].astype(int)
    out = np.zeros_like(td)
    for c in range(k):
        ref = np.sort(cd[cl == c])
        m = tl == c
        if not m.any():
            continue
        if len(ref) < 20:  # too few calibration patches: fall back to the global null
            ref = np.sort(cd.ravel())
        d = td[m]
        if variant == 'z':
            out[m] = (d - ref.mean()) / (ref.std() + 1e-8)
        elif variant == 'robust':
            q1, q2, q3 = np.percentile(ref, [25, 50, 75])
            out[m] = (d - q2) / (q3 - q1 + 1e-8)
        elif variant == 'conf':
            n = len(ref)
            ge = n - np.searchsorted(ref, d, side='left')
            p = (1 + ge) / (n + 1.0)
            tail = ref[int(0.99 * n):]                  # exponential tail above the calib max
            scale = max(tail.mean() - tail[0], 1e-6)
            beyond = d > ref[-1]
            p[beyond] = (1.0 / (n + 1.0)) * np.exp(-(d[beyond] - ref[-1]) / scale)
            out[m] = -np.log10(p)
    return out


def main():
    cache = {}
    variants = ['raw', 'z', 'robust', 'conf']
    rows = []
    for rep in REPS:
        for v in variants:
            for k in (KS if v != 'raw' else [0]):
                ps = patch_scores(rep, k, v)
                cache[(rep, v, k)] = ps
                m = metrics(ps.max(1))
                rows.append(((rep,), v, k, m))
    # fusion of the two parts of the proposed detector, same variant and K
    for v in variants:
        for k in (KS if v != 'raw' else [0]):
            if v == 'raw':  # raw distances of different blocks are on different scales: use calib z per image
                continue
            ps = (cache[('L8', v, k)] + cache[('L10_pool3', v, k)]) / 2
            cache[(('L8', 'L10_pool3'), v, k)] = ps
            rows.append((('L8', 'L10_pool3'), v, k, metrics(ps.max(1))))
    print('%-16s %-7s %3s %6s %6s %6s %6s %6s' % ('rep', 'variant', 'K', 'AUROC', 'AUPRC', 'R@1%', 'Rsmall', 'Rmed'))
    for rep, v, k, m in rows:
        print('%-16s %-7s %3d %6.3f %6.3f %6.3f %6.3f %6.3f' % ('+'.join(rep), v, k, *m))

    # nested selection over (variant, K) for the fused detector
    cands = [(('L8', 'L10_pool3'), v, k) for v in variants[1:] for k in KS]
    rng = np.random.RandomState(0); idn, ida = np.arange(nn), np.arange(nn, len(y)); outs, ch = [], {}
    for rep_i in range(10):
        pn, pa = rng.permutation(idn), rng.permutation(ida)
        H = [np.r_[pn[: nn // 2], pa[: len(pa) // 2]], np.r_[pn[nn // 2:], pa[len(pa) // 2:]]]
        for f in (0, 1):
            best = max(cands, key=lambda c: metrics(cache[c].max(1), H[1 - f])[[0, 2]].sum())
            key = '%s K=%d' % (best[1], best[2]); ch[key] = ch.get(key, 0) + 1
            outs.append(metrics(cache[best].max(1), H[f]))
    o = np.array(outs)
    print('\nnested selection over %d (variant, K) of the fused detector:' % len(cands))
    print('AUROC %.3f+-%.3f  AUPRC %.3f+-%.3f  R@1%% %.3f+-%.3f  Rsmall %.3f+-%.3f  Rmed %.3f+-%.3f'
          % tuple(v for mm, ss in zip(o.mean(0), o.std(0)) for v in (mm, ss)))
    print('chosen:', ch)
    json.dump([{'rep': '+'.join(r), 'variant': v, 'k': k, 'auroc': m[0], 'auprc': m[1], 'r1': m[2],
                'rsmall': m[3], 'rmed': m[4]} for r, v, k, m in rows],
              open(R + 'outputs/structcal/results.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
