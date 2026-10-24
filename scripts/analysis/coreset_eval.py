"""Greedy coreset vs random subsampling vs exhaustive bank, for the baseline representation
(final-layer tokens) and for the proposed detector (block 8 + pooled block 10, fused).

Reads outputs/coreset_<rep>/raw/{greedy,random}_k<K>.npz (scripts/run_coreset.py) and the
exhaustive banks outputs/bigbank*/raw/n23688.npz (scripts/run_bigbank.py).
Image score = max of the patch map; fused score = mean of the two calibration z-scores.
Reports AUROC, Sensitivity@(Specificity = 0.99) (threshold at the 99th percentile of the
normal test scores, as in the paper), small-defect (<0.1 %) sensitivity, and a paired
bootstrap 95 % CI of the difference to the exhaustive bank.

numpy + PIL only:  python3 -I scripts/analysis/coreset_eval.py
"""
import glob
import json
import re

import numpy as np
from PIL import Image

R = '/autofs/space/crater_001/projects/micropath/semi/'
EXH = {'L12': 'bigbank', 'L8': 'bigbank_L8', 'L10_pool3': 'bigbank_L10_pool3'}

abn = sorted(p for p in glob.glob(R + 'MIIC/Anomaly_test/abnormal_img/*') if p.lower().endswith(('.jpg', '.png')))
nrm = sorted(p for p in glob.glob(R + 'MIIC/Anomaly_test/normal_img/*') if p.lower().endswith(('.jpg', '.png')))
nn = len(nrm)
areas = np.array([(np.asarray(Image.open(R + 'MIIC/Anomaly_test/abnormal_mask/%s_mask.jpg'
                                         % p.split('/')[-1].rsplit('.', 1)[0]).convert('L')) > 127).mean() * 100 for p in abn])
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


def zscore(path):
    z = np.load(path)
    t = z['test_map'].astype(np.float32).reshape(len(z['test_map']), -1).max(1)
    c = z['calib_map'].astype(np.float32).reshape(len(z['calib_map']), -1).max(1)
    return (t - c.mean()) / c.std()


def budgets(rep):
    return sorted({int(re.search(r'_k(\d+)\.npz', f).group(1)) for f in glob.glob(R + 'outputs/coreset_%s/raw/*.npz' % rep)})


def boot_diff(a, b, n=2000):
    rng = np.random.RandomState(0); i0, i1 = np.arange(nn), np.arange(nn, len(y)); d = []
    for _ in range(n):
        idx = np.r_[rng.choice(i0, nn), rng.choice(i1, len(i1))]
        d.append(metrics(a, idx) - metrics(b, idx))
    return np.percentile(np.array(d), [2.5, 97.5], axis=0)


def report(name, scores, ref):
    out = []
    print('\n' + name)
    print('%-22s %7s %8s %8s   %s' % ('bank', 'AUROC', 'Sens99', 'Small', '95% CI of Sens99 / Small diff. to exhaustive'))
    m0 = metrics(ref)
    print('%-22s %7.3f %8.3f %8.3f' % ('exhaustive (32.4 M)', *m0))
    for lab, s in scores:
        m = metrics(s); lo, hi = boot_diff(s, ref)
        print('%-22s %7.3f %8.3f %8.3f   [%+.3f, %+.3f] / [%+.3f, %+.3f]' % (lab, *m, lo[1], hi[1], lo[2], hi[2]))
        out.append({'bank': lab, 'auroc': m[0], 'sens_spec99': m[1], 'sens_small': m[2],
                    'ci_diff_sens': [lo[1], hi[1]], 'ci_diff_small': [lo[2], hi[2]]})
    return {'exhaustive': dict(zip(('auroc', 'sens_spec99', 'sens_small'), m0)), 'rows': out}


def main():
    res = {}
    if budgets('L12'):  # baseline representation: raw max distance, as in the diagnosis table
        raw = lambda p: (lambda z: z.reshape(len(z), -1).max(1))(np.load(p)['test_map'].astype(np.float32))
        sc = [('%s %d' % (m, k), raw(R + 'outputs/coreset_L12/raw/%s_k%d.npz' % (m, k)))
              for k in budgets('L12') for m in ('greedy', 'random')]
        res['L12'] = report('Final-layer tokens (baseline representation)', sc, raw(R + 'outputs/bigbank/raw/n23688.npz'))
    if budgets('L8') and budgets('L10_pool3'):
        ks = sorted(set(budgets('L8')) & set(budgets('L10_pool3')))
        ref = (zscore(R + 'outputs/bigbank_L8/raw/n23688.npz') + zscore(R + 'outputs/bigbank_L10_pool3/raw/n23688.npz')) / 2
        sc = [('%s %d' % (m, k), (zscore(R + 'outputs/coreset_L8/raw/%s_k%d.npz' % (m, k))
                                  + zscore(R + 'outputs/coreset_L10_pool3/raw/%s_k%d.npz' % (m, k))) / 2)
              for k in ks for m in ('greedy', 'random')]
        res['fused'] = report('Proposed detector (block 8 + pooled block 10, fused)', sc, ref)
    json.dump(res, open(R + 'outputs/coreset_results.json', 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
