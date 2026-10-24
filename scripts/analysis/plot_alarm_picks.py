"""Figure: the first normal images that alarm-driven acquisition adds to the memory (MIIC).

Reads outputs/acquire_alarm/acquired.json (scripts/run_acquire.py, initial memory 1) and shows
the 16 highest-scoring pool images of the first round, with their fused z-scores.  The pool
order is rebuilt exactly as in run_acquire.py (calibration split seed 42, bank_seed 0).

    uv run python scripts/analysis/plot_alarm_picks.py
"""
import json
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_patchcore import list_images  # noqa: E402

R = Path('/autofs/space/crater_001/projects/micropath/semi')
OUT = R / 'outputs/report/assets/alarm_picks.png'
FONT = '/usr/share/fonts/dejavu/DejaVuSans.ttf'


def main():
    train = list_images(sorted((R / 'MIIC').glob('Anomaly_train/train_normal_*.jpg')))
    perm = torch.randperm(len(train), generator=torch.Generator().manual_seed(42)).tolist()
    pool = [train[j] for j in perm[200:]]
    order = torch.randperm(len(pool), generator=torch.Generator().manual_seed(0)).tolist()
    pool = [pool[j] for j in order]
    r1 = next(x for x in json.load(open(R / 'outputs/acquire_alarm/acquired.json')) if x['round'] == 1)
    n_col, n_row, t, pad, head = 8, 2, 300, 8, 34
    fig = Image.new('RGB', (n_col * (t + pad) + pad, n_row * (t + head + pad) + pad), 'white')
    d = ImageDraw.Draw(fig); f = ImageFont.truetype(FONT, 22)
    for k, (i, z) in enumerate(list(zip(r1['added'], r1['scores']))[: n_col * n_row]):
        x, y = pad + (k % n_col) * (t + pad), pad + (k // n_col) * (t + head + pad)
        fig.paste(Image.open(pool[i]).convert('RGB').resize((t, t), Image.BILINEAR), (x, y + head))
        d.text((x + 2, y + 4), '%d   z = %.1f' % (k + 1, z), font=f, fill='black')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.save(OUT)
    print('saved ->', OUT, [pool[i].name for i in r1['added'][:4]])


if __name__ == '__main__':
    main()
