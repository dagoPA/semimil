"""Figure: which normal images set the 1% FPR threshold, and which small defects fall below it.

Top row: the highest-scoring normal test images under the baseline detector (final-layer
DINOv2 tokens, 100k random-token bank, seed 0).  Bottom row: small defects (<0.1 % of the
image) that this detector misses at the 1 % false-positive operating point, with the
ground-truth contour, and whether the proposed detector (block 8 + pooled block 10,
exhaustive banks) catches them at its own 1 % operating point.

numpy + PIL only (runs on the login node):  python3 -I scripts/analysis/plot_threshold_diagnosis.py
Writes outputs/diagnostics/threshold_diagnosis.png (6 x 2) and a compact 4 x 2 teaser
for the manuscript (figs/threshold_diagnosis.png).
"""
import glob

import numpy as np
from PIL import Image, ImageDraw, ImageFont

R = '/autofs/space/crater_001/projects/micropath/semi/'
OUTS = [R + 'outputs/diagnostics/threshold_diagnosis.png',
        R + 'outputs/report/latex/figs/threshold_diagnosis.png']
N_COL, T, PAD, HEAD = 6, 240, 8, 30
FONT = '/usr/share/fonts/dejavu/DejaVuSans.ttf'
BOLD = '/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf'


def images(sub):
    return sorted(p for p in glob.glob(R + f'MIIC/Anomaly_test/{sub}/*') if p.lower().endswith(('.jpg', '.png')))


def zscores(run):
    z = np.load(R + f'outputs/{run}/raw/n23688.npz')
    t = z['test_map'].astype(np.float32).reshape(len(z['test_map']), -1).max(1)
    c = z['calib_map'].astype(np.float32).reshape(len(z['calib_map']), -1).max(1)
    return (t - c.mean()) / c.std()


def contour(mask):
    m = mask.astype(bool)
    p = np.pad(m, 1, mode='constant')
    inner = p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
    return m & ~inner


def tile(path, label, mask_path=None, color=(255, 255, 255)):
    im = Image.open(path).convert('L').resize((T, T), Image.BILINEAR).convert('RGB')
    if mask_path:
        m = np.asarray(Image.open(mask_path).convert('L').resize((T, T), Image.NEAREST)) > 127
        e = contour(m)
        for _ in range(2):  # thicken the contour so tiny defects stay visible
            e = e | np.roll(e, 1, 0) | np.roll(e, 1, 1)
        a = np.asarray(im).copy()
        a[e] = (0, 220, 0)
        im = Image.fromarray(a)
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, T, 22), fill=(0, 0, 0))
    d.text((5, 3), label, font=ImageFont.truetype(FONT, 14), fill=color)
    return im


def main():
    norm, abn = images('normal_img'), images('abnormal_img')
    nn = len(norm)
    base = np.load(R + 'outputs/tiled/raw/g1_518_s0.npz')['test_score']
    thr_b = np.percentile(base[:nn], 99)
    prop = (zscores('bigbank_L8') + zscores('bigbank_L10_pool3')) / 2
    thr_p = np.percentile(prop[:nn], 99)
    masks = [R + 'MIIC/Anomaly_test/abnormal_mask/%s_mask.jpg' % p.split('/')[-1].rsplit('.', 1)[0] for p in abn]
    areas = np.array([(np.asarray(Image.open(m).convert('L')) > 127).mean() * 100 for m in masks])

    top = np.argsort(-base[:nn])[:N_COL]
    missed = [i for i in np.argsort(-base[nn:]) if areas[i] < 0.1 and base[nn + i] < thr_b]
    caught = [i for i in missed if prop[nn + i] >= thr_p]
    pick = (caught + [i for i in missed if i not in caught])[:N_COL]  # show recoveries first

    W = N_COL * (T + PAD) + PAD
    H = 2 * (HEAD + T + PAD) + PAD
    fig = Image.new('RGB', (W, H), 'white')
    d = ImageDraw.Draw(fig)
    bold = ImageFont.truetype(BOLD, 16)
    d.text((PAD, 6), '(a) Highest-scoring normal test images under the baseline detector (seed 0); '
           'they set its 1%% false-positive threshold (%.1f)' % thr_b, font=bold, fill='black')
    for j, i in enumerate(top):
        fig.paste(tile(norm[i], 'normal image, score %.1f' % base[i]), (PAD + j * (T + PAD), HEAD))
    y0 = HEAD + T + PAD
    d.text((PAD, y0 + 6), '(b) Small defects (<0.1%% of the image) missed by the baseline detector; '
           'green: ground truth. %d of %d are caught by the proposed detector'
           % (len(caught), len(missed)), font=bold, fill='black')
    for j, i in enumerate(pick):
        ok = prop[nn + i] >= thr_p
        lab = '%.3f%% area: %s' % (areas[i], 'caught' if ok else 'still missed')
        fig.paste(tile(abn[i], lab, masks[i], (120, 255, 120) if ok else (255, 160, 160)),
                  (PAD + j * (T + PAD), y0 + HEAD))
    fig.save(OUTS[0])

    # compact single-column teaser for the manuscript (first page): 4 x 2 tiles
    n, t, pad, head = 4, 300, 10, 44
    tf = ImageFont.truetype(BOLD, 22)
    W2, H2 = n * (t + pad) + pad, 2 * (head + t + pad) + pad
    fig2 = Image.new('RGB', (W2, H2), 'white')
    d2 = ImageDraw.Draw(fig2)
    d2.text((pad, 10), '(a) Normal images that set the threshold', font=tf, fill='black')
    for j, i in enumerate(top[:n]):
        im = tile(norm[i], 'normal, %.1f > %.1f' % (base[i], thr_b))
        fig2.paste(im.resize((t, t), Image.BILINEAR), (pad + j * (t + pad), head))
    y1 = head + t + pad
    d2.text((pad, y1 + 10), '(b) Small defects missed by the baseline', font=tf, fill='black')
    still = [i for i in missed if i not in caught]
    show = caught[:n - 1] + still[:1]  # not only successes: one defect is still missed
    for j, i in enumerate(show):
        ok = prop[nn + i] >= thr_p
        im = tile(abn[i], '%.3f%%: %s' % (areas[i], 'caught' if ok else 'missed'), masks[i],
                  (120, 255, 120) if ok else (255, 160, 160))
        fig2.paste(im.resize((t, t), Image.BILINEAR), (pad + j * (t + pad), y1 + head))
    fig2.save(OUTS[1])
    print('threshold baseline %.2f, missed small %d, caught by proposed %d' % (thr_b, len(missed), len(caught)))


if __name__ == '__main__':
    main()
