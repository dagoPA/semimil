"""MIIC evaluation following Metrics Reloaded (Maier-Hein et al., Nat. Methods 2024).

Problem fingerprint. (1) Image-level binary classification (ImLC): defective vs. normal,
strong class imbalance; the test prevalence (116/1,388 = 8.4 %) does not reflect a
production line (0.46 % in MIIC overall); the decision rule is target-value based
(a 1 % false-alarm budget, i.e. specificity 0.99); scores are not probabilities, so no
calibration metric is requested.  Metrics: AUROC (prevalence-independent multi-threshold
metric), Sensitivity@(Specificity = 0.99) and the positive likelihood ratio LR+ (per-class
counting metrics robust to prevalence), balanced accuracy and MCC at that operating point,
and the full confusion matrix; AP is reported only as a secondary, prevalence-dependent
metric.  95 % CIs by stratified bootstrap (2,000 resamples).
(2) Localization: defects are small (median equivalent side 15 px) and several per image,
so it is framed as object detection (ObD).  Detections are local maxima of each anomaly
map; localization criterion: the peak lies within 16 px (one ViT token) of a reference
region (point inside the dilated mask); assignment: greedy by score, double assignments
ignored; metrics: FROC score (mean sensitivity at 1/8, 1/4, 1/2, 1, 2, 4 and 8 false
positives per image, over all 1,388 test images) and region-level AP.

numpy + scipy + PIL only:  python3 -I scripts/analysis/metrics_reloaded.py
Writes outputs/metrics_reloaded/miic.json.
"""
import csv
import glob
import json

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

R = '/autofs/space/crater_001/projects/micropath/semi/'
OUT = R + 'outputs/metrics_reloaded/'
RES = 256                      # map resolution for detection
TOL = 16 * RES // 512          # 16 px at native resolution
FPPI = [0.125, 0.25, 0.5, 1, 2, 4, 8]
B = 2000


# ----------------------------------------------------------------------------- data
nrm = sorted(p for p in glob.glob(R + 'MIIC/Anomaly_test/normal_img/*') if p.lower().endswith(('.jpg', '.png')))
abn = sorted(p for p in glob.glob(R + 'MIIC/Anomaly_test/abnormal_img/*') if p.lower().endswith(('.jpg', '.png')))
nn = len(nrm)
y = np.r_[np.zeros(nn, int), np.ones(len(abn), int)]
names = [p.split('/')[-1] for p in nrm + abn]
masks512 = [np.asarray(Image.open(R + 'MIIC/Anomaly_test/abnormal_mask/%s_mask.jpg' % p.split('/')[-1].rsplit('.', 1)[0])
                       .convert('L')) > 127 for p in abn]
# reference regions (8-connected) at 512 px; tolerance zone and area per region at RES
regions = []  # (image index, dilated zone at RES, area % of image)
for k, m in enumerate(masks512):
    lab, n = ndi.label(m, structure=np.ones((3, 3)))
    for r in range(1, n + 1):
        reg = lab == r
        zone = np.asarray(Image.fromarray((reg * 255).astype(np.uint8)).resize((RES, RES), Image.NEAREST)) > 0
        if not zone.any():  # tiny region lost by resizing: keep its centroid
            cy, cx = ndi.center_of_mass(reg)
            zone = np.zeros((RES, RES), bool); zone[int(cy * RES / 512), int(cx * RES / 512)] = True
        zone = ndi.binary_dilation(zone, iterations=TOL)
        regions.append((nn + k, zone, reg.mean() * 100))


# ----------------------------------------------------------------------------- metrics
def auroc(yy, s):
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    r = (np.cumsum(cnt) - (cnt - 1) / 2.0)[inv]
    n1 = (yy == 1).sum(); n0 = len(yy) - n1
    return (r[yy == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def ap(yy, s):
    o = np.argsort(-s, kind='mergesort'); t = yy[o]; tp = np.cumsum(t)
    return float((tp / np.arange(1, len(t) + 1) * t).sum() / t.sum())


def at_spec(yy, s, spec=0.99):
    thr = np.percentile(s[yy == 0], 100 * spec)
    pred = s > thr
    tp = int((pred & (yy == 1)).sum()); fn = int((~pred & (yy == 1)).sum())
    fp = int((pred & (yy == 0)).sum()); tn = int((~pred & (yy == 0)).sum())
    sens, sp = tp / (tp + fn), tn / (tn + fp)
    mcc_den = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return {'sens': sens, 'spec': sp, 'lr_plus': sens / max(1 - sp, 1e-9), 'ba': (sens + sp) / 2,
            'mcc': (tp * tn - fp * fn) / mcc_den if mcc_den else 0.0, 'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn}


def image_metrics(s):
    d = at_spec(y, s)
    d.update({'auroc': float(auroc(y, s)), 'ap': ap(y, s)})
    return d


def bootstrap_ci(s, seed=0):
    rng = np.random.RandomState(seed)
    i0, i1 = np.where(y == 0)[0], np.where(y == 1)[0]
    vals = []
    for _ in range(B):
        idx = np.r_[rng.choice(i0, len(i0)), rng.choice(i1, len(i1))]
        yy, ss = y[idx], s[idx]
        a = at_spec(yy, ss)
        vals.append([auroc(yy, ss), ap(yy, ss), a['sens'], a['ba'], a['mcc']])
    v = np.array(vals)
    lo, hi = np.percentile(v, 2.5, 0), np.percentile(v, 97.5, 0)
    return {k: [float(lo[j]), float(hi[j])] for j, k in enumerate(['auroc', 'ap', 'sens', 'ba', 'mcc'])}


def detections(maps, top=20):
    """Local maxima of each map (after a light smoothing) -> list of (image, y, x, score)."""
    out = []
    for i, m in enumerate(maps):
        m = ndi.gaussian_filter(m.astype(np.float32), 2)
        peaks = (m == ndi.maximum_filter(m, size=17)) & (m > np.percentile(m, 50))
        ys, xs = np.nonzero(peaks)
        if len(ys) > top:
            keep = np.argsort(-m[ys, xs])[:top]; ys, xs = ys[keep], xs[keep]
        out += [(i, yy, xx, float(m[yy, xx])) for yy, xx in zip(ys, xs)]
    return out


def detection_metrics(dets):
    by_img = {}
    for j, (img, zone, area) in enumerate(regions):
        by_img.setdefault(img, []).append(j)
    order = sorted(dets, key=lambda t: -t[3])
    matched = np.zeros(len(regions), bool)
    tps, fps = [], []
    for img, yy, xx, sc in order:
        hit = [j for j in by_img.get(img, []) if regions[j][1][yy, xx]]
        if not hit:
            tps.append(0); fps.append(1)
        elif all(matched[j] for j in hit):
            continue                                   # double assignment: ignored
        else:
            matched[[j for j in hit if not matched[j]][0]] = True; tps.append(1); fps.append(0)
    tp, fp = np.cumsum(tps), np.cumsum(fps)
    sens, fppi = tp / len(regions), fp / len(y)
    froc = [float(sens[fppi <= f].max()) if (fppi <= f).any() else 0.0 for f in FPPI]
    prec = tp / np.maximum(tp + fp, 1)
    t = np.array(tps)
    ap_det = float((prec * t).sum() / len(regions))
    # sensitivity at 1 FP per image, by region size
    k1 = np.where(fppi <= 1)[0]
    det_at1 = np.zeros(len(regions), bool)
    if len(k1):
        cut = sorted(dets, key=lambda q: -q[3])  # re-match up to the 1-FPPI cut
        matched2 = np.zeros(len(regions), bool); nfp = 0
        for img, yy, xx, sc in cut:
            hit = [j for j in by_img.get(img, []) if regions[j][1][yy, xx]]
            if not hit:
                nfp += 1
                if nfp / len(y) > 1:
                    break
            elif not all(matched2[j] for j in hit):
                matched2[[j for j in hit if not matched2[j]][0]] = True
        det_at1 = matched2
    areas = np.array([r[2] for r in regions])
    small = areas < 0.1
    return {'froc_score': float(np.mean(froc)), 'froc': dict(zip(map(str, FPPI), froc)), 'ap_det': ap_det,
            'sens_at_1fppi': float(det_at1.mean()), 'sens_small_at_1fppi': float(det_at1[small].mean()),
            'n_regions': len(regions), 'n_small_regions': int(small.sum())}


# ----------------------------------------------------------------------------- methods
def up(maps):
    out = np.empty((len(maps), RES, RES), np.float32)
    for i, m in enumerate(maps):
        out[i] = np.asarray(Image.fromarray(m.astype(np.float32)).resize((RES, RES), Image.BILINEAR))
    return out


def proposed():
    zs, zm = [], []
    for run in ('bigbank_L8', 'bigbank_L10_pool3'):
        z = np.load(R + 'outputs/%s/raw/n23688.npz' % run)
        t, c = z['test_map'].astype(np.float32), z['calib_map'].astype(np.float32)
        ti, ci = t.reshape(len(t), -1).max(1), c.reshape(len(c), -1).max(1)
        zs.append((ti - ci.mean()) / ci.std()); zm.append((t - c.mean()) / c.std())
    yield np.mean(zs, 0), up(np.mean(zm, 0))


def baseline():
    for s in (0, 1, 2):
        z = np.load(R + 'outputs/tiled/raw/g1_518_s%d.npz' % s)
        yield z['test_score'].astype(np.float64), up(z['test_map'].astype(np.float32))


def sota(path):
    for d in sorted(glob.glob(R + 'outputs/sota/%s/seed*/' % path)):
        sc = {r['name']: float(r['score']) for r in csv.DictReader(open(d + 'image_scores.csv'))}
        yield np.array([sc[n] for n in names]), np.load(d + 'maps256.npy', mmap_mode='r')


METHODS = [('Proposed', proposed), ('Baseline (100k bank)', baseline),
           ('Dinomaly', lambda: sota('dinomaly/sep_it5000_r448')),
           ('SubspaceAD (all normals)', lambda: sota('subspacead/fullshot_all')),
           ('INP-Former', lambda: sota('inp_former/single_ep200_n1000_r448')),
           ('AnomalyDINO (1000 normals)', lambda: sota('anomalydino/1000shot')),
           ('GLASS', lambda: sota('glass/custom_ep640_n1000'))]


def main():
    import os
    os.makedirs(OUT, exist_ok=True)
    res = {}
    for name, fn in METHODS:
        img, det, first = [], [], None
        for s, m in fn():  # one run at a time
            img.append(image_metrics(s)); det.append(detection_metrics(detections(m)))
            first = s if first is None else first
        runs = [(first, None)] * len(img)
        keys_i = ['auroc', 'ap', 'sens', 'lr_plus', 'ba', 'mcc']
        keys_d = ['froc_score', 'ap_det', 'sens_at_1fppi', 'sens_small_at_1fppi']
        r = {'n_runs': len(runs),
             'image_mean': {k: float(np.mean([m[k] for m in img])) for k in keys_i},
             'image_std': {k: float(np.std([m[k] for m in img])) for k in keys_i},
             'image_ci_run0': bootstrap_ci(runs[0][0]),
             'confusion_run0': {k: img[0][k] for k in ('tp', 'fn', 'fp', 'tn')},
             'det_mean': {k: float(np.mean([m[k] for m in det])) for k in keys_d},
             'det_std': {k: float(np.std([m[k] for m in det])) for k in keys_d},
             'froc_run0': det[0]['froc']}
        res[name] = r
        print('%-28s n=%d AUROC %.3f [%.3f,%.3f] Sens@Sp.99 %.3f [%.3f,%.3f] LR+ %.1f BA %.3f MCC %.3f AP %.3f | '
              'FROC %.3f APdet %.3f Sens@1FPPI %.3f small %.3f'
              % (name, len(runs), r['image_mean']['auroc'], *r['image_ci_run0']['auroc'], r['image_mean']['sens'],
                 *r['image_ci_run0']['sens'], r['image_mean']['lr_plus'], r['image_mean']['ba'], r['image_mean']['mcc'],
                 r['image_mean']['ap'], r['det_mean']['froc_score'], r['det_mean']['ap_det'],
                 r['det_mean']['sens_at_1fppi'], r['det_mean']['sens_small_at_1fppi']), flush=True)
    res['_meta'] = {'n_regions': len(regions), 'n_small_regions': int(sum(r[2] < 0.1 for r in regions)),
                    'prevalence_test': float(y.mean()), 'tolerance_px_native': 16, 'fppi': FPPI, 'bootstrap': B}
    json.dump(res, open(OUT + 'miic.json', 'w'), indent=1)
    print(res['_meta'])


if __name__ == '__main__':
    main()
