#!/usr/bin/env python
"""Collect outputs/sota/<method>/<setting>/seed*/metrics.json into one table (mean +/- std over seeds).

std is the sample std (ddof=1); single-run settings show no std.  Writes summary.csv / summary.md
under outputs/sota.  Light: reads JSON only.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT = Path("/autofs/space/crater_001/projects/micropath/semi/outputs/sota")
KEYS = ["image_auroc", "image_auprc", "pixel_auroc", "pro", "recall_1pct_fpr",
        "recall_small", "recall_medium", "recall_large"]


def fmt(v: list[float]) -> str:
    a = np.array(v, float)
    return f"{a.mean():.3f}" if len(a) == 1 else f"{a.mean():.3f} ± {a.std(ddof=1):.3f}"


def main() -> None:
    groups = defaultdict(list)
    for f in sorted(OUT.glob("*/*/seed*/metrics.json")):
        if f.parts[-3].startswith("SMOKE"):
            continue
        groups[(f.parts[-4], f.parts[-3])].append(json.loads(f.read_text()))
    rows = []
    for (method, setting), ms in sorted(groups.items()):
        r = {"method": method, "setting": setting, "seeds": len(ms),
             "n_train": ms[0].get("run_info", {}).get("n_train_images")}
        for k in KEYS:
            r[k] = fmt([m[k] for m in ms])
        tt = [m.get("run_info", {}).get("train_time_s", np.nan) for m in ms]
        r["train_time_min"] = fmt([t / 60 for t in tt])
        rows.append(r)
    with (OUT / "summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    hdr = list(rows[0])
    lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    lines += ["| " + " | ".join(str(r[h]) for h in hdr) + " |" for r in rows]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
