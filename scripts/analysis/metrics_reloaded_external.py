"""Metrics Reloaded-style evaluation of the external datasets (Carinthia-S, Texture-AD, MVTec AD).

Per task (Carinthia-S fold, Texture-AD subclass x photograph fold, MVTec AD category):
image-level AUROC, AP and Sensitivity@(Specificity = 0.95) (the 0.99 point is not
estimable with the few normals per task); detection of reference regions as in
metrics_reloaded.py (peaks of the anomaly map within 16 px of a region at 512 px,
greedy by score, double assignments ignored; FROC score over FPPI 1/8..8 and
sensitivity at 1 FPPI).  Aggregation respects the hierarchy: metrics per task, then
mean over tasks, with a 95 % bootstrap CI over tasks.
Detectors: baseline (final-layer tokens, 100k-token bank) and proposed (exhaustive
banks; image score = calibrated fusion of block 8 and pooled block 10; localization
from the block-8 map).

numpy + scipy + PIL only:  python3 -I scripts/analysis/metrics_reloaded_external.py
Writes outputs/metrics_reloaded/external.json.
"""
import csv
import glob
import json
import re

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

R = '/autofs/space/crater_001/projects/micropath/semi/outputs/'
RES = 128
TOL = 16 * RES // 512
FPPI = [0.125, 0.25, 0.5, 1, 2, 4, 8]
CROP = {'carinthia': 8, 'texturead': 0, 'mvtec': 0}
RUNS = {  # dataset -> (baseline run dir, proposed run dir)
    'carinthia': ('external/run', 'external/run_full'),
    'texturead': ('external/run', 'external/run_full'),
    'mvtec': ('external/run_mvtec_base', 'external/run_mvtec_full'),
}


def auroc(yy, s):
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    r = (np.cumsum(cnt) - (cnt - 1) / 2.0)[inv]
    n1 = (yy == 1).sum(); n0 = len(yy) - n1
    return (r[yy == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def ap(yy, s):
    o = np.argsort(-s, kind='mergesort'); t = yy[o]; tp = np.cumsum(t)
    return float((tp / np.arange(1, len(t) + 1) * t).sum() / t.sum())


def mask_regions(path, crop):
    if not path:
        return []
    m = Image.open(path).convert('L')
    if crop:
        m = m.crop((crop, crop, m.width - crop, m.height - crop))
    m = np.asarray(m.resize((512, 512), Image.NEAREST)) > 127
    lab, n = ndi.label(m, structure=np.ones((3, 3)))
    zones = []
    for r in range(1, n + 1):
        reg = lab == r
        z = np.asarray(Image.fromarray((reg * 255).astype(np.uint8)).resize((RES, RES), Image.NEAREST)) > 0
        if not z.any():
            cy, cx = ndi.center_of_mass(reg)
            z = np.zeros((RES, RES), bool); z[int(cy * RES / 512), int(cx * RES / 512)] = True
        zones.append(ndi.binary_dilation(z, iterations=TOL))
    return zones


def detect(maps, top=20):
    out = []
    for i, m in enumerate(maps):
        m = ndi.gaussian_filter(m.astype(np.float32), 1)
        pk = (m == ndi.maximum_filter(m, size=9)) & (m > np.percentile(m, 50))
        ys, xs = np.nonzero(pk)
        if len(ys) > top:
            k = np.argsort(-m[ys, xs])[:top]; ys, xs = ys[k], xs[k]
        out += [(i, a, b, float(m[a, b])) for a, b in zip(ys, xs)]
    return out


def froc(dets, zones_per_img, n_img):
    regs = [(i, z) for i, zs in enumerate(zones_per_img) for z in zs]
    if not regs:
        return float('nan'), float('nan')
    by = {}
    for j, (i, z) in enumerate(regs):
        by.setdefault(i, []).append(j)
    matched = np.zeros(len(regs), bool); tps, fps = [], []
    for i, a, b, s in sorted(dets, key=lambda t: -t[3]):
        hit = [j for j in by.get(i, []) if regs[j][1][a, b]]
        if not hit:
            tps.append(0); fps.append(1)
        elif all(matched[j] for j in hit):
            continue
        else:
            matched[[j for j in hit if not matched[j]][0]] = True; tps.append(1); fps.append(0)
    tp, fp = np.cumsum(tps), np.cumsum(fps)
    sens, fppi = tp / len(regs), fp / n_img
    pts = [float(sens[fppi <= f].max()) if (fppi <= f).any() else 0.0 for f in FPPI]
    return float(np.mean(pts)), pts[FPPI.index(1)]


def load(run, ds, ext, task):
    return np.load(R + '%s/raw/%s/%s__g1_518/%s_s0.npz' % (run, ds, ext, task))


def zscore(z):
    c = z['calib_score']
    return (z['test_score'] - c.mean()) / c.std()


def task_metrics(y, score, maps, zones):
    n0 = (y == 0).sum()
    thr = np.percentile(score[y == 0], 95)
    return {'auroc': float(auroc(y, score)), 'ap': ap(y, score), 'sens_spec95': float((score[y == 1] > thr).mean()),
            'n_normal': int(n0), **dict(zip(('froc', 'sens_1fppi'), froc(detect(maps), zones, len(y))))}


def summarize(rows, key):
    v = np.array([r[key] for r in rows if r[key] == r[key]])
    rng = np.random.RandomState(0)
    bs = [rng.choice(v, len(v)).mean() for _ in range(2000)]
    return {'mean': float(v.mean()), 'std': float(v.std()), 'ci': [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}


def main():
    out = {}
    for ds, (base_run, prop_run) in RUNS.items():
        rows = list(csv.DictReader(open(R + 'external/%s.csv' % ds)))
        tasks = sorted({re.sub(r'_s0\.npz$', '', f.split('/')[-1])
                        for f in glob.glob(R + '%s/raw/%s/frozen_L8__g1_518/*_s0.npz' % (prop_run, ds))})
        res = {'baseline': [], 'proposed': []}
        for t in tasks:
            zb = load(base_run, ds, 'frozen', t)
            idx = zb['test_idx']
            y = np.array([int(rows[i]['label']) for i in idx])
            zones = [mask_regions(rows[i]['mask'], CROP[ds]) for i in idx]
            res['baseline'].append(task_metrics(y, zb['test_score'], zb['test_map'], zones))
            z8, z10 = load(prop_run, ds, 'frozen_L8', t), load(prop_run, ds, 'frozen_L10_pool3', t)
            assert (z8['test_idx'] == idx).all()
            res['proposed'].append(task_metrics(y, (zscore(z8) + zscore(z10)) / 2, z8['test_map'], zones))
        out[ds] = {m: {k: summarize(v, k) for k in ('auroc', 'ap', 'sens_spec95', 'froc', 'sens_1fppi')}
                   for m, v in res.items()}
        out[ds]['n_tasks'] = len(tasks)
        out[ds]['per_task'] = {m: dict(zip(tasks, v)) for m, v in res.items()}
        for m in ('baseline', 'proposed'):
            s = out[ds][m]
            print('%-10s %-9s AUROC %.3f [%.3f,%.3f]  AP %.3f  Sens@Sp.95 %.3f [%.3f,%.3f]  FROC %.3f [%.3f,%.3f]  Sens@1FPPI %.3f'
                  % (ds, m, s['auroc']['mean'], *s['auroc']['ci'], s['ap']['mean'], s['sens_spec95']['mean'],
                     *s['sens_spec95']['ci'], s['froc']['mean'], *s['froc']['ci'], s['sens_1fppi']['mean']), flush=True)
    json.dump(out, open(R + 'metrics_reloaded/external.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
