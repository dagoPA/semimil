"""Nested (cross-fitted) label-free fusion analysis on MIIC; numpy only, runs on the login node.

fuse_nested_layers.py: 100k-bank layer study (outputs/layers, outputs/tiled).
fuse_nested_fullbank.py: exhaustive banks (outputs/bigbank*). Usage: python3 -I <script>
"""
"""Offline label-free fusion of single-scale runs (same calibration set and seeds)."""
import sys, glob, itertools, numpy as np
from PIL import Image
R = '/autofs/space/crater_001/projects/micropath/semi/'
def auroc(y, s):
    y = np.asarray(y); s = np.asarray(s); _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    r = (np.cumsum(cnt) - (cnt - 1) / 2.0)[inv]; n1 = (y == 1).sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)
def ap(y, s):
    o = np.argsort(-s, kind='mergesort'); y = np.asarray(y)[o]; tp = np.cumsum(y); prec = tp / np.arange(1, len(y) + 1)
    return (prec * y).sum() / y.sum()
abn = sorted(p for p in glob.glob(R + 'MIIC/Anomaly_test/abnormal_img/*') if p.lower().endswith(('.jpg', '.png')))
areas = np.array([(np.asarray(Image.open(R + 'MIIC/Anomaly_test/abnormal_mask/%s_mask.jpg' % p.split('/')[-1].rsplit('.', 1)[0]).convert('L')) > 127).mean() * 100 for p in abn])
nn = 1272; y = np.r_[np.zeros(nn), np.ones(len(abn))]
runs = {'L12': R + 'outputs/tiled/raw/g1_518_s%d.npz'}
for d in sorted(glob.glob(R + 'outputs/layers/*/raw/g1_518_s0.npz')):
    runs[d.split('/')[-3]] = d.replace('_s0.npz', '_s%d.npz')
def metrics(score):
    n, a = score[:nn], score[nn:]; thr = np.percentile(n, 99)
    out = [auroc(y, score), ap(y, score), (a >= thr).mean()]
    for lo, hi in [(0, .1), (.1, 1)]:
        sel = (areas >= lo) & (areas < hi); out.append((a[sel] >= thr).mean())
    return np.array(out)
def z(name, s):
    d = np.load(runs[name] % s); c = d['calib_score']; return (d['test_score'] - c.mean()) / c.std()
combos = [(k,) for k in runs] + [c for c in itertools.combinations(sorted(runs), 2) if 'L12' in c or any('pool' in x for x in c)]
res = []
for c in combos:
    m = np.mean([metrics(np.mean([z(k, s) for k in c], 0)) for s in (0, 1, 2)], 0); res.append((c, m))
res.sort(key=lambda t: -(t[1][2] + t[1][0]))
print('%-28s %6s %6s %6s %6s %6s' % ('combo', 'AUROC', 'AUPRC', 'R@1%', 'Rsmall', 'Rmed'))
for c, m in res[:int(sys.argv[1]) if len(sys.argv) > 1 else 25]:
    print('%-28s %6.3f %6.3f %6.3f %6.3f %6.3f' % ('+'.join(c), *m))

# ---- pre-specified pairs: same layer(s) with and without local averaging ----
print('\npre-specified (same features, raw + pooled):')
for c in [('L12', 'L12_pool3'), ('L8', 'L8_pool3'), ('L6-9-12', 'L6-9-12_pool3'), ('L10_pool3',), ('L12',)]:
    if all(k in runs for k in c):
        m = np.mean([metrics(np.mean([z(k, s) for k in c], 0)) for s in (0, 1, 2)], 0)
        print('%-28s %6.3f %6.3f %6.3f %6.3f %6.3f' % ('+'.join(c), *m))

# ---- nested (cross-fitted) selection: choose on one half of the test set, score on the other ----
rng = np.random.RandomState(0)
idx_n, idx_a = np.arange(nn), np.arange(nn, nn + len(abn))
cache = {(c, s): np.mean([z(k, s) for k in c], 0) for c in combos for s in (0, 1, 2)}
def fold_metrics(score, sel_idx):
    yy = y[sel_idx]; sc = score[sel_idx]; n = sc[yy == 0]; a = sc[yy == 1]; thr = np.percentile(n, 99)
    ar = areas[sel_idx[yy == 1] - nn]
    return np.array([auroc(yy, sc), ap(yy, sc), (a >= thr).mean(), (a[ar < .1] >= thr).mean()])
outs, chosen = [], {}
for rep in range(10):
    pn, pa = rng.permutation(idx_n), rng.permutation(idx_a)
    halves = [np.r_[pn[: nn // 2], pa[: len(pa) // 2]], np.r_[pn[nn // 2:], pa[len(pa) // 2:]]]
    for f in (0, 1):
        sel, ev = halves[1 - f], halves[f]
        best = max(combos, key=lambda c: np.mean([fold_metrics(cache[(c, s)], sel)[[0, 2]].sum() for s in (0, 1, 2)]))
        chosen['+'.join(best)] = chosen.get('+'.join(best), 0) + 1
        outs.append(np.mean([fold_metrics(cache[(best, s)], ev) for s in (0, 1, 2)], 0))
o = np.array(outs)
print('\nnested selection over %d candidates (criterion AUROC + R@1%%, 10 x 2 folds):' % len(combos))
print('AUROC %.3f+-%.3f  AUPRC %.3f+-%.3f  R@1%% %.3f+-%.3f  Rsmall %.3f+-%.3f' % tuple(v for m, s in zip(o.mean(0), o.std(0)) for v in (m, s)))
base = np.array([np.mean([fold_metrics(cache[(('L12',), s)], h) for s in (0, 1, 2)], 0) for h in [np.r_[rng.permutation(idx_n)[: nn // 2], rng.permutation(idx_a)[: 58]] for _ in range(20)]])
print('chosen:', dict(sorted(chosen.items(), key=lambda t: -t[1])))
