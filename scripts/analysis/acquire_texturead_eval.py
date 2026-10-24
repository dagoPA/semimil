"""Alarm-driven vs random acquisition on Texture-AD: paired analysis over tasks.

Reads outputs/acquire_texturead/part_*/results.json (scripts/run_acquire_texturead.py).  A task
is one (wafer subclass, photograph fold).  For every task, seed and memory size the difference
alarm minus random is taken (both strategies start from the same initial memory); differences
are averaged over seeds, then summarized over tasks (each task weighs the same):
  - per memory fraction: mean difference with a 95 % bootstrap CI over tasks;
  - over the whole learning curve (all sizes after the shared initial one): mean difference,
    bootstrap CI, and a two-sided Wilcoxon signed-rank test over tasks;
  - memory needed to reach the random strategy's final AUROC (at the largest fraction).
Also writes the learning-curve figure outputs/report/assets/acquisition_texturead.{png,pdf}.

    uv run python scripts/analysis/acquire_texturead_eval.py
"""
import glob
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
from scipy.stats import wilcoxon

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

R = Path('/autofs/space/crater_001/projects/micropath/semi/outputs')
METRICS = ('auroc', 'sens95', 'sens99')


def main():
    rows = [r for f in sorted(glob.glob(str(R / 'acquire_texturead/part_*/results.json'))) for r in json.load(open(f))]
    v = defaultdict(dict)  # (task, seed, strategy) -> {memory index: row}
    sizes = defaultdict(set)
    for r in rows:
        t = (r['cls'], r['fold'])
        sizes[t].add(r['memory'])
    for r in rows:
        t = (r['cls'], r['fold'])
        v[(t, r['seed'], r['strategy'])][sorted(sizes[t]).index(r['memory'])] = r
    tasks = sorted(sizes, key=lambda t: (int(t[0]), t[1]))
    seeds = sorted({r['seed'] for r in rows})
    n_idx = min(len(sizes[t]) for t in tasks)
    fr = [np.mean([v[(t, seeds[0], 'random')][i]['fraction'] for t in tasks]) for i in range(n_idx)]
    # D[task, index, metric]: alarm - random, averaged over seeds
    D = np.array([[[np.mean([v[(t, s, 'alarm')][i][m] - v[(t, s, 'random')][i][m] for s in seeds
                             if i in v[(t, s, 'alarm')] and i in v[(t, s, 'random')]]) for m in METRICS]
                   for i in range(n_idx)] for t in tasks])
    A = {st: np.array([[[np.mean([v[(t, s, st)][i][m] for s in seeds]) for m in METRICS] for i in range(n_idx)]
                       for t in tasks]) for st in ('alarm', 'random')}
    rng = np.random.RandomState(0)
    boot = np.array([D[rng.choice(len(tasks), len(tasks))].mean(0) for _ in range(5000)])
    print('%d tasks (%d defective test crops), seeds %s' % (len(tasks), sum(v[(t, seeds[0], 'random')][0]['n_defect'] for t in tasks), seeds))
    print('%8s  %-30s %-30s %-30s' % ('memory', 'AUROC  alarm-random [95% CI]', 'Sens@Spec0.95', 'Sens@Spec0.99'))
    for i in range(n_idx):
        cells = []
        for k in range(3):
            lo, hi = np.percentile(boot[:, i, k], [2.5, 97.5])
            cells.append('%+.4f [%+.4f, %+.4f]' % (D[:, i, k].mean(), lo, hi))
        print('%7.0f%%  %s' % (100 * fr[i], '  '.join(cells)))
    curve = D[:, 1:, :].mean(1)  # per task, average over the curve
    bc = boot[:, 1:, :].mean(1)
    out = {'tasks': len(tasks), 'fractions': fr, 'mean_diff': D.mean(0).tolist(), 'curve': {}}
    for k, m in enumerate(METRICS):
        lo, hi = np.percentile(bc[:, k], [2.5, 97.5])
        p = wilcoxon(curve[:, k]).pvalue if np.any(curve[:, k] != 0) else 1.0
        wins = int((curve[:, k] > 0).sum())
        print('curve average %-7s %+.4f [%+.4f, %+.4f]  Wilcoxon p = %.4f  alarm better in %d of %d tasks'
              % (m, curve[:, k].mean(), lo, hi, p, wins, len(tasks)))
        out['curve'][m] = {'mean': float(curve[:, k].mean()), 'ci': [float(lo), float(hi)], 'p': float(p), 'wins': wins}
    # memory needed to reach the random strategy's final AUROC
    need = {'alarm': [], 'random': []}
    for j, t in enumerate(tasks):
        target = A['random'][j, -1, 0] - 1e-9
        for st in need:
            idx = next((i for i in range(n_idx) if A[st][j, i, 0] >= target), None)
            need[st].append(fr[idx] if idx is not None else np.nan)
    na, nr = np.array(need['alarm']), np.array(need['random'])
    print('memory to reach the random final AUROC: alarm %.1f%% (median), random %.1f%%; alarm earlier in %d, later in %d, tie %d tasks'
          % (100 * np.nanmedian(na), 100 * np.nanmedian(nr), int((na < nr).sum()), int((na > nr).sum()), int((na == nr).sum())))
    json.dump(out, open(R / 'acquire_texturead_results.json', 'w'), indent=1)

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.3))
    for ax, k, lab in ((axes[0], 0, 'Image AUROC'), (axes[1], 1, 'Sensitivity at specificity 0.95'),
                       (axes[2], 2, 'Sensitivity at specificity 0.99')):
        for st, c, mk in (('alarm', '#c53030', 'o'), ('random', '#8c8c8c', 's')):
            ax.plot(100 * np.array(fr), A[st][:, :, k].mean(0), marker=mk, ms=4, color=c,
                    label='alarm-driven' if st == 'alarm' else 'random')
        ax.set_xlabel('memory (% of the normal pool)'); ax.set_title(lab, fontsize=10); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    p = R / 'report/assets/acquisition_texturead.png'
    fig.savefig(p, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(p.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    print('saved ->', p)


if __name__ == '__main__':
    main()
