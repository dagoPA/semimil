"""Fusion of exhaustive-bank runs (one bank, deterministic) with nested 2-fold selection."""
import glob, itertools, numpy as np
from PIL import Image
exec(open(__file__.replace('fuse_nested_fullbank.py', 'fuse_nested_layers.py')).read().split("runs = {")[0])  # auroc, ap, areas, y, nn
R = '/autofs/space/crater_001/projects/micropath/semi/'
names = {'L12': 'bigbank', 'L8': 'bigbank_L8', 'L8_pool3': 'bigbank_L8_pool3', 'L10_pool3': 'bigbank_L10_pool3'}
Z = {}
for k, d in names.items():
    z = np.load(R + f'outputs/{d}/raw/n23688.npz')
    t = z['test_map'].astype(np.float32).reshape(len(z['test_map']), -1).max(1)
    c = z['calib_map'].astype(np.float32).reshape(len(z['calib_map']), -1).max(1)
    Z[k] = (t - c.mean()) / c.std()
def met(score, idx=None):
    idx = np.arange(len(y)) if idx is None else idx
    yy, sc = y[idx], score[idx]; n, a = sc[yy == 0], sc[yy == 1]; thr = np.percentile(n, 99); ar = areas[idx[yy == 1] - nn]
    return np.array([auroc(yy, sc), ap(yy, sc), (a >= thr).mean(), (a[ar < .1] >= thr).mean(), (a[(ar >= .1) & (ar < 1)] >= thr).mean()])
combos = [(k,) for k in Z] + list(itertools.combinations(Z, 2)) + list(itertools.combinations(Z, 3))
print('%-26s %6s %6s %6s %6s %6s' % ('full bank, combo', 'AUROC', 'AUPRC', 'R@1%', 'Rsmall', 'Rmed'))
for c in combos:
    print('%-26s %6.3f %6.3f %6.3f %6.3f %6.3f' % ('+'.join(c), *met(np.mean([Z[k] for k in c], 0))))
rng = np.random.RandomState(0); idn, ida = np.arange(nn), np.arange(nn, len(y)); outs, ch = [], {}
for rep in range(10):
    pn, pa = rng.permutation(idn), rng.permutation(ida)
    H = [np.r_[pn[: nn // 2], pa[: len(pa) // 2]], np.r_[pn[nn // 2:], pa[len(pa) // 2:]]]
    for f in (0, 1):
        best = max(combos, key=lambda c: met(np.mean([Z[k] for k in c], 0), H[1 - f])[[0, 2]].sum())
        ch['+'.join(best)] = ch.get('+'.join(best), 0) + 1
        outs.append(met(np.mean([Z[k] for k in best], 0), H[f]))
o = np.array(outs)
print('\nnested selection (full banks, %d candidates):' % len(combos))
print('AUROC %.3f+-%.3f  AUPRC %.3f+-%.3f  R@1%% %.3f+-%.3f  Rsmall %.3f+-%.3f  Rmed %.3f+-%.3f' % tuple(v for m, s in zip(o.mean(0), o.std(0)) for v in (m, s)))
print('chosen:', ch)
