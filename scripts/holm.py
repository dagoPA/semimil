#!/usr/bin/env python
"""Holm-Bonferroni correction for the DeLong comparisons vs the reference method.

Reads outputs/stats/delong.json (m paired AUROC tests vs A-DINOv2), computes the
Holm step-down adjusted p-values, and writes them back as `p_holm`. Holm controls
the family-wise error rate without assuming independence -- appropriate here since
all comparisons share the reference method.

    uv run python scripts/holm.py
"""

from __future__ import annotations

import json
from pathlib import Path

STATS = Path("/autofs/space/crater_001/projects/micropath/semi/outputs/stats")


def holm(pvals: list[float]) -> list[float]:
    """Holm step-down adjusted p-values, preserving input order."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * pvals[idx])
        running = max(running, val)  # enforce monotonicity
        adj[idx] = running
    return adj


def main() -> None:
    delong = json.loads((STATS / "delong.json").read_text())
    comps = delong["comparisons"]
    padj = holm([c["p_delong"] for c in comps])
    for c, pa in zip(comps, padj, strict=True):
        c["p_holm"] = pa
    delong["correction"] = "holm-bonferroni"
    delong["n_comparisons"] = len(comps)
    (STATS / "delong.json").write_text(json.dumps(delong, indent=2))

    print(f"Holm-Bonferroni (m={len(comps)}) vs {delong['reference']}:")
    print(f"  {'method':<24}{'p_DeLong':>12}{'p_Holm':>12}  sig(0.05)")
    for c in sorted(comps, key=lambda x: x["p_delong"]):
        sig = "yes" if c["p_holm"] < 0.05 else "n.s."
        print(f"  {c['method']:<24}{c['p_delong']:>12.2e}{c['p_holm']:>12.2e}  {sig}")


if __name__ == "__main__":
    main()
