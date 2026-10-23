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

W, H = 1500, 600
def img(k):
    return 'data:image/png;base64,' + base64.b64encode(CROPS[k]).decode()
GRAY, BLUE, RED, GREEN = '#9a9a9a', '#2b6cb0', '#c53030', '#2f855a'
o = []
a = o.append
a(f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{W}" height="{H}" '
  f'viewBox="0 0 {W} {H}" font-family="DejaVu Sans, sans-serif">')
a('<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">'
  '<path d="M0,0 L10,5 L0,10 z" fill="#444"/></marker>'
  '<style>.h{font-size:26px;font-weight:bold;fill:#111}.t{font-size:20px;fill:#111;text-anchor:middle}'
  '.s{font-size:17px;fill:#444;text-anchor:middle}.ln{stroke:#444;stroke-width:2;fill:none;marker-end:url(#a)}'
  '.box{stroke:#444;stroke-width:1.5}</style></defs>')
a(f'<rect width="{W}" height="{H}" fill="white"/>')
# panel separators
for x in (500, 1000):
    a(f'<line x1="{x}" y1="20" x2="{x}" y2="{H-20}" stroke="#ccc" stroke-width="2"/>')

# ---- panel 1: diagnosis
a('<text class="h" x="20" y="44">Why small defects are missed</text>')
a(f'<image x="20" y="70" width="220" height="220" xlink:href="{img("normal")}"/>')
a(f'<image x="260" y="70" width="220" height="220" xlink:href="{img("defect")}"/>')
a(f'<rect x="20" y="70" width="220" height="220" fill="none" stroke="{RED}" stroke-width="4"/>')
a(f'<rect x="260" y="70" width="220" height="220" fill="none" stroke="{GREEN}" stroke-width="4"/>')
a('<text class="t" x="130" y="318">rare normal</text>')
a(f'<text class="s" x="130" y="342" fill="{RED}" style="fill:{RED}">false alarm</text>')
a('<text class="t" x="370" y="318">particle (0.06 %)</text>')
a(f'<text class="s" x="370" y="342" style="fill:{GREEN}">missed by the baseline</text>')
# score axis
y0, x_lo, x_hi, v_lo, v_hi = 440, 40, 460, 16.5, 20.5
X = lambda v: x_lo + (v - v_lo) / (v_hi - v_lo) * (x_hi - x_lo)
a(f'<line x1="{x_lo}" y1="{y0}" x2="{x_hi}" y2="{y0}" stroke="#444" stroke-width="2"/>')
a(f'<line x1="{X(17.6):.0f}" y1="{y0-45}" x2="{X(17.6):.0f}" y2="{y0+30}" stroke="#111" stroke-width="3" stroke-dasharray="7,5"/>')
a(f'<text class="s" x="{X(17.6):.0f}" y="{y0-54}">1 % false-alarm threshold</text>')
a(f'<circle cx="{X(19.7):.0f}" cy="{y0}" r="10" fill="{RED}"/>')
a(f'<circle cx="{X(17.3):.0f}" cy="{y0}" r="10" fill="{GREEN}"/>')
a(f'<text class="s" x="{(x_lo+x_hi)/2:.0f}" y="{y0+58}">anomaly score of the baseline detector</text>')
a('<text class="s" x="250" y="545">Rare normal structures</text>')
a('<text class="s" x="250" y="568">set the threshold.</text>')

# ---- panel 2: detector
cx = 750
a(f'<text class="h" x="{cx}" y="44" text-anchor="middle">Detector: no labels, no training</text>')
def box(x, y, w, h, fill, t1, t2=None):
    a(f'<rect class="box" x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}"/>')
    if t2:
        a(f'<text class="t" x="{x+w/2}" y="{y+h/2-4}">{t1}</text><text class="s" x="{x+w/2}" y="{y+h/2+20}">{t2}</text>')
    else:
        a(f'<text class="t" x="{x+w/2}" y="{y+h/2+7}">{t1}</text>')
box(560, 70, 380, 60, '#f2f2f2', 'Frozen DINOv2 ViT-S/14', 'patch tokens of the test image')
a('<path class="ln" d="M700,130 L645,165"/><path class="ln" d="M800,130 L855,165"/>')
box(520, 168, 220, 72, '#dbe8f6', 'Block 8', 'sharp: small defects')
box(760, 168, 220, 72, '#fbe5d0', 'Block 10, 3×3 avg', 'smooth: quiet normals')
a('<path class="ln" d="M630,240 L630,275"/><path class="ln" d="M870,240 L870,275"/>')
a('<rect class="box" x="520" y="278" width="460" height="100" rx="8" fill="#f2f2f2"/>')
a('<text class="t" x="750" y="310">Exhaustive memory, exact search</text>')
a('<text class="s" x="750" y="340">every token of 23,688 normal images</text>')
a('<text class="s" x="750" y="362">(32.4 M per block, one GPU)</text>')
a('<path class="ln" d="M750,378 L750,413"/>')
box(560, 416, 380, 72, '#e6f2e6', 'Calibrated fusion', 'z-scores on 200 normals')
a('<text class="s" x="750" y="545">Covering rare normals lowers the threshold;</text>')
a('<text class="s" x="750" y="568">sharp tokens lift tiny particles above it.</text>')

# ---- panel 3: results
a('<text class="h" x="1250" y="44" text-anchor="middle">Results</text>')
def bars(x0, y_top, h, title, groups, vmax=1.0):
    a(f'<text class="t" x="{x0+210}" y="{y_top-12}">{title}</text>')
    base_y = y_top + h
    a(f'<line x1="{x0}" y1="{base_y}" x2="{x0+430}" y2="{base_y}" stroke="#444" stroke-width="1.5"/>')
    gw = 430 / len(groups)
    for k, (lab, b, p) in enumerate(groups):
        gx = x0 + k * gw + gw / 2
        for j, (v, c) in enumerate(((b, GRAY), (p, BLUE))):
            bh = v / vmax * h
            bx = gx - 46 + j * 48
            a(f'<rect x="{bx:.0f}" y="{base_y-bh:.0f}" width="44" height="{bh:.0f}" fill="{c}"/>')
            a(f'<text x="{bx+22:.0f}" y="{base_y-bh-6:.0f}" font-size="16" text-anchor="middle" '
              f'fill="{c if j else "#555"}" font-weight="{"bold" if j else "normal"}">{v:.2f}</text>')
        a(f'<text class="s" x="{gx:.0f}" y="{base_y+22}">{lab}</text>')
bars(1035, 130, 120, 'MIIC: sensitivity at specificity 0.99',
     [('all defects', 0.787, 0.940), ('smallest (&lt;0.1 %)', 0.52, 0.841)])
bars(1035, 365, 120, 'Detection FROC, applied unchanged',
     [('SEM', 0.738, 0.950), ('wafer', 0.614, 0.709), ('MVTec AD', 0.820, 0.883)])
a(f'<rect x="1090" y="530" width="22" height="22" fill="{GRAY}"/><text x="1120" y="548" font-size="18" fill="#333">baseline</text>')
a(f'<rect x="1260" y="530" width="22" height="22" fill="{BLUE}"/><text x="1290" y="548" font-size="18" fill="#333">proposed</text>')
a('</svg>')
open(R + 'outputs/report/latex/graphical_abstract.svg', 'w').write('\n'.join(o))
