#!/usr/bin/env python
"""Look up authoritative DOIs for the paper's references via the Crossref API.

Prints, per reference, the matched title + DOI so it can be eyeballed before
pasting into the LaTeX. Venues that do not mint DOIs (NeurIPS, ICML/PMLR, TMLR)
fall back to their arXiv DOI (10.48550/arXiv.<id>), which is a valid DOI.

    uv run python scripts/lookup_dois.py
"""

from __future__ import annotations

import json
import ssl
import urllib.parse
import urllib.request

import certifi

# uv Python has no system CA bundle -> give urllib an explicit certifi context
SSL_CTX = ssl.create_default_context(cafile=certifi.where())
MAILTO = "dpulido@mgh.harvard.edu"

# (bibkey, title, arxiv_fallback_id or None)
REFS = [
    ("miic", "Joint anomaly detection and inpainting for microscopy images via deep self-supervised learning", None),
    ("patchcore", "Towards total recall in industrial anomaly detection", None),
    ("padim", "PaDiM a patch distribution modeling framework for anomaly detection and localization", None),
    ("dino", "Emerging properties in self-supervised vision transformers", None),
    ("dinov2", "DINOv2 learning robust visual features without supervision", "2304.07193"),
    ("byol", "Bootstrap your own latent a new approach to self-supervised learning", "2006.07733"),
    ("ilse", "Attention-based deep multiple instance learning", "1802.04712"),
    ("mvtec", "MVTec AD a comprehensive real-world dataset for unsupervised anomaly detection", None),
    ("draem", "DRAEM a discriminatively trained reconstruction embedding for surface anomaly detection", None),
    ("cutpaste", "CutPaste self-supervised learning for anomaly detection and localization", None),
    ("transmil", "TransMIL transformer based correlated multiple instance learning for whole slide image classification", "2106.00908"),
    ("mae", "Masked autoencoders are scalable vision learners", None),
    ("wrn", "Wide residual networks", "1605.07146"),
    ("barusco", "Evaluating modern visual anomaly detection approaches in semiconductor manufacturing a comparative study", "2505.07576"),
    ("fastflow", "FastFlow unsupervised anomaly detection and localization via 2D normalizing flows", "2111.07677"),
    ("stfpm", "Student-teacher feature pyramid matching for anomaly detection", "2103.04257"),
    ("cfa", "CFA coupled-hypersphere-based feature adaptation for target-oriented anomaly localization", None),
    ("anomalib", "Anomalib a deep learning library for anomaly detection", None),
]


def crossref(title: str) -> tuple[str, str, str] | None:
    q = urllib.parse.urlencode({"query.bibliographic": title, "rows": 1, "mailto": MAILTO})
    url = f"https://api.crossref.org/works?{q}"
    try:
        with urllib.request.urlopen(url, timeout=25, context=SSL_CTX) as r:
            items = json.loads(r.read())["message"]["items"]
        if not items:
            return None
        it = items[0]
        return it.get("DOI", ""), " ".join(it.get("title", [""])), it.get("type", "")
    except Exception as e:  # noqa: BLE001
        return None


def main() -> None:
    print(f"{'key':<11} DOI / matched title")
    print("-" * 90)
    out = {}
    for key, title, arxiv in REFS:
        res = crossref(title)
        if res:
            doi, mtitle, mtype = res
            out[key] = doi
            print(f"{key:<11} {doi}")
            print(f"{'':<11} -> {mtitle[:76]}  [{mtype}]")
        else:
            out[key] = None
            print(f"{key:<11} (crossref: no result)")
        if arxiv:
            print(f"{'':<11} arXiv fallback: 10.48550/arXiv.{arxiv}")
    print("\n== quick dict ==")
    print(json.dumps(out, indent=0))


if __name__ == "__main__":
    main()
