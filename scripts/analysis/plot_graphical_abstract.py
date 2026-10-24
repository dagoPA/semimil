"""Graphical abstract for the journal submission (Elsevier: >= 1328 x 531 px, readable at 13 x 5 cm).

Left: the highest-scoring normal test image under the baseline detector (it sets the 1 %
false-alarm threshold) and the first small defect (<0.1 % of the image) that the baseline
misses and the proposed detector catches, with their positions on the baseline score axis.
Middle: the detector.  Right: headline results (numbers from final/metrics.json and
metrics_reloaded/external.json).

numpy + PIL only:  python3 -I scripts/analysis/plot_graphical_abstract.py
Writes outputs/report/latex/graphical_abstract.svg (images embedded); render with
  rsvg-convert -z 2 graphical_abstract.svg -o graphical_abstract.png
  rsvg-convert -f pdf graphical_abstract.svg -o graphical_abstract.pdf
and copy the PNG to docs/graphical_abstract.png for the README.
"""
import base64
import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_threshold_diagnosis as P
R = P.R
norm, abn = P.images('normal_img'), P.images('abnormal_img')
nn = len(norm)
base = np.load(R + 'outputs/tiled/raw/g1_518_s0.npz')['test_score']
thr_b = np.percentile(base[:nn], 99)
prop = (P.zscores('bigbank_L8') + P.zscores('bigbank_L10_pool3')) / 2
thr_p = np.percentile(prop[:nn], 99)
masks = [R + 'MIIC/Anomaly_test/abnormal_mask/%s_mask.jpg' % p.split('/')[-1].rsplit('.', 1)[0] for p in abn]
areas = np.array([(np.asarray(Image.open(m).convert('L')) > 127).mean() * 100 for m in masks])
top = np.argsort(-base[:nn])
missed = [i for i in np.argsort(-base[nn:]) if areas[i] < 0.1 and base[nn + i] < thr_b]
caught = [i for i in missed if prop[nn + i] >= thr_p]
S = 400
CROPS = {}
def save(path, out, mask=None):
    im = Image.open(path).convert('L').resize((S, S), Image.BILINEAR).convert('RGB')
    if mask:
        m = np.asarray(Image.open(mask).convert('L').resize((S, S), Image.NEAREST)) > 127
        e = P.contour(m)
        for _ in range(3):
            e = e | np.roll(e, 1, 0) | np.roll(e, 1, 1)
        a = np.asarray(im).copy(); a[e] = (0, 220, 0); im = Image.fromarray(a)
    buf = io.BytesIO(); im.save(buf, 'PNG'); CROPS[out] = buf.getvalue()
save(norm[top[0]], 'normal')
i = caught[0]
save(abn[i], 'defect', masks[i])
print('normal score %.1f thr %.1f; defect area %.3f%% base %.1f prop %.2f thr_p %.2f'
      % (base[top[0]], thr_b, areas[i], base[nn + i], prop[nn + i], thr_p))

W, H = 1500, 600   # text >= 28 px so that it stays legible at 13 cm wide
def img(k):
    return 'data:image/png;base64,' + base64.b64encode(CROPS[k]).decode()
GRAY, BLUE, RED, GREEN, INK = '#8c8c8c', '#2b6cb0', '#c53030', '#2f855a', '#111'
o = []
a = o.append
a(f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{W}" height="{H}" '
  f'viewBox="0 0 {W} {H}" font-family="DejaVu Sans, sans-serif">')
a('<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto">'
  '<path d="M0,0 L10,5 L0,10 z" fill="#444"/></marker>'
  '<style>.h{font-size:40px;font-weight:bold;fill:#111;text-anchor:middle}'
  '.t{font-size:31px;fill:#111;text-anchor:middle}.s{font-size:28px;fill:#333;text-anchor:middle}'
  '.ln{stroke:#444;stroke-width:3;fill:none;marker-end:url(#a)}.box{stroke:#444;stroke-width:2}</style></defs>')
a(f'<rect width="{W}" height="{H}" fill="white"/>')
for x in (500, 1000):
    a(f'<line x1="{x}" y1="20" x2="{x}" y2="{H-20}" stroke="#ccc" stroke-width="3"/>')

# ---- panel 1: the problem
a('<text class="h" x="250" y="50">Problem</text>')
a(f'<image x="30" y="80" width="205" height="205" xlink:href="{img("normal")}"/>')
a(f'<image x="265" y="80" width="205" height="205" xlink:href="{img("defect")}"/>')
a(f'<rect x="30" y="80" width="205" height="205" fill="none" stroke="{RED}" stroke-width="6"/>')
a(f'<rect x="265" y="80" width="205" height="205" fill="none" stroke="{GREEN}" stroke-width="6"/>')
a(f'<text class="t" x="132" y="325" style="fill:{RED}">rare normal</text>')
a(f'<text class="t" x="367" y="325" style="fill:{GREEN}">tiny defect</text>')
y0, x_lo, x_hi, v_lo, v_hi = 440, 40, 460, 16.6, 20.4   # baseline scores on a line
X = lambda v: x_lo + (v - v_lo) / (v_hi - v_lo) * (x_hi - x_lo)
a(f'<line x1="{x_lo}" y1="{y0}" x2="{x_hi}" y2="{y0}" stroke="#444" stroke-width="3"/>')
a(f'<line x1="{X(17.6):.0f}" y1="{y0-40}" x2="{X(17.6):.0f}" y2="{y0+30}" stroke="#111" stroke-width="4" stroke-dasharray="9,6"/>')
a(f'<text class="s" x="{X(17.6):.0f}" y="{y0-52}">alarm threshold</text>')
a(f'<circle cx="{X(19.7):.0f}" cy="{y0}" r="14" fill="{RED}"/>')
a(f'<circle cx="{X(17.3):.0f}" cy="{y0}" r="14" fill="{GREEN}"/>')
a(f'<text class="s" x="250" y="{y0+65}">baseline anomaly score</text>')
a('<text class="s" x="250" y="565" style="fill:#111">Rare normals set the threshold</text>')

# ---- panel 2: the detector
a('<text class="h" x="750" y="50">Detector</text>')
def box(x, y, w, h, fill, t1, t2=None):
    a(f'<rect class="box" x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill}"/>')
    if t2:
        a(f'<text class="t" x="{x+w/2}" y="{y+h/2-6}">{t1}</text><text class="s" x="{x+w/2}" y="{y+h/2+28}">{t2}</text>')
    else:
        a(f'<text class="t" x="{x+w/2}" y="{y+h/2+11}">{t1}</text>')
box(585, 78, 330, 62, '#f2f2f2', 'Frozen DINOv2')
a('<path class="ln" d="M700,140 L640,172"/><path class="ln" d="M800,140 L860,172"/>')
box(525, 176, 215, 62, '#dbe8f6', 'sharp block')
box(760, 176, 215, 62, '#fbe5d0', 'smooth block')
a('<path class="ln" d="M632,238 L632,268"/><path class="ln" d="M868,238 L868,268"/>')
box(525, 272, 450, 100, '#f2f2f2', 'Covers rare normals', 'all tokens or a 1 % coreset')
a('<path class="ln" d="M750,372 L750,402"/>')
box(585, 406, 330, 62, '#e6f2e6', 'Calibrated fusion')
a('<text class="s" x="750" y="565" style="fill:#111">No labels, no training</text>')

# ---- panel 3: results (baseline gray, proposed blue)
a('<text class="h" x="1250" y="50">Results</text>')
def bars(y_title, base_y, title, groups, scale=95):
    a(f'<text class="s" x="1250" y="{y_title}">{title}</text>')
    a(f'<line x1="1030" y1="{base_y}" x2="1470" y2="{base_y}" stroke="#444" stroke-width="2"/>')
    gw = 440 / len(groups)
    for k, (lab, b, p) in enumerate(groups):
        gx = 1030 + k * gw + gw / 2
        for j, (v, c) in enumerate(((b, GRAY), (p, BLUE))):
            bh = v * scale
            bx = gx - 56 + j * 58
            a(f'<rect x="{bx:.0f}" y="{base_y-bh:.0f}" width="54" height="{bh:.0f}" fill="{c}"/>')
            if j:
                a(f'<text x="{bx+27:.0f}" y="{base_y-bh-8:.0f}" font-size="28" font-weight="bold" '
                  f'text-anchor="middle" fill="{BLUE}">{v:.2f}</text>')
        a(f'<text class="s" x="{gx:.0f}" y="{base_y+32}">{lab}</text>')
bars(100, 240, 'MIIC sensitivity',
     [('all defects', 0.787, 0.940), ('smallest', 0.52, 0.841)])
bars(330, 470, 'Detection on new datasets',
     [('SEM', 0.738, 0.950), ('wafer', 0.614, 0.709), ('MVTec', 0.820, 0.883)])
a(f'<rect x="1050" y="540" width="28" height="28" fill="{GRAY}"/><text x="1088" y="565" font-size="28" fill="#333">baseline</text>')
a(f'<rect x="1250" y="540" width="28" height="28" fill="{BLUE}"/><text x="1288" y="565" font-size="28" fill="#333">proposed</text>')
a('</svg>')
open(R + 'outputs/report/latex/graphical_abstract.svg', 'w').write('\n'.join(o))
