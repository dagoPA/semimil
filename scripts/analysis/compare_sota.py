import csv, glob, numpy as np
from PIL import Image
exec(open('/homes/8/dp003/code/development/semi/scripts/analysis/fuse_nested_layers.py').read().split("runs = {")[0])
R = '/autofs/space/crater_001/projects/micropath/semi/'
norm = sorted(p for p in glob.glob(R + 'MIIC/Anomaly_test/normal_img/*') if p.lower().endswith(('.jpg', '.png')))
names = [p.split('/')[-1] for p in norm] + [p.split('/')[-1] for p in abn]
Z = []
for d in ('bigbank_L8', 'bigbank_L10_pool3'):
    z = np.load(R + f'outputs/{d}/raw/n23688.npz')
    t = z['test_map'].astype(np.float32).reshape(len(z['test_map']), -1).max(1); c = z['calib_map'].astype(np.float32).reshape(len(z['calib_map']), -1).max(1)
    Z.append((t - c.mean()) / c.std())
ours = np.mean(Z, 0)
def load(path):
    d = {r['name']: float(r['score']) for r in csv.DictReader(open(path))}
    return np.array([d[n] for n in names])
def met(s):
    n, a = s[:nn], s[nn:]; thr = np.percentile(n, 99)
    return np.array([auroc(y, s), ap(y, s), (a >= thr).mean(), (a[areas < .1] >= thr).mean()])
meth = {'Dinomaly': 'dinomaly/sep_it5000_r448', 'INP-Former': 'inp_former/single_ep200_n1000_r448', 'SubspaceAD (all normals)': 'subspacead/fullshot_all',
        'AnomalyDINO (1000)': 'anomalydino/1000shot', 'GLASS': 'glass/custom_ep640_n1000'}
print('%-26s %5s %14s %14s %14s %14s' % ('method', 'seeds', 'AUROC', 'AUPRC', 'R@1%FPR', 'R small'))
print('%-26s %5s %14.3f %14.3f %14.3f %14.3f' % ('Ours (L8 + L10 pooled)', '-', *met(ours)))
rng = np.random.RandomState(0); B = 2000
idx_n, idx_a = np.arange(nn), np.arange(nn, len(y))
boots = [np.r_[rng.choice(idx_n, nn), rng.choice(idx_a, len(idx_a))] for _ in range(B)]
for m, p in meth.items():
    fs = sorted(glob.glob(R + f'outputs/sota/{p}/seed*/image_scores.csv'))
    ms = np.array([met(load(f)) for f in fs])
    cell = ['%.3f+-%.3f' % (a, b) for a, b in zip(ms.mean(0), ms.std(0))]
    print('%-26s %5d %14s %14s %14s %14s' % (m, len(fs), *cell))
    if m in ('Dinomaly', 'INP-Former', 'SubspaceAD (all normals)'):
        other = np.mean([ (lambda s: (s - s[:nn].mean()) / s[:nn].std())(load(f)) for f in fs], 0)  # seed-averaged, rank-preserving per seed
        o1 = load(fs[0])
        diffs = []
        for b in boots:
            yy = y[b]; sa, sb = ours[b], o1[b]
            def rec(s):
                thr = np.percentile(s[yy == 0], 99); return (s[yy == 1] >= thr).mean()
            diffs.append([auroc(yy, sa) - auroc(yy, sb), ap(yy, sa) - ap(yy, sb), rec(sa) - rec(sb)])
        d = np.array(diffs)
        for k, lab in enumerate(['AUROC', 'AUPRC', 'R@1%']):
            lo, hi = np.percentile(d[:, k], [2.5, 97.5]); pval = 2 * min((d[:, k] <= 0).mean(), (d[:, k] >= 0).mean())
            print('    vs %s seed0, delta %s: %+.3f [%.3f, %.3f] p=%.3f' % (m, lab, d[:, k].mean(), lo, hi, pval))
