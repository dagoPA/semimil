#!/usr/bin/env python
"""Build the self-contained article HTML from paper.src.html.

Replaces __IMG_*__ tokens with base64-embedded JPEGs and entity-encodes every
non-ASCII character (fixes charset issues when the artifact host guesses the
encoding). Figures live as PNGs in assets/ and are compressed to assets/web/*.jpg
on the way in.

    uv run python scripts/build_paper.py
"""

from __future__ import annotations

import base64
from pathlib import Path

from PIL import Image

REPORT = Path("/autofs/space/crater_001/projects/micropath/semi/outputs/report")
SRC = REPORT / "paper.src.html"
OUT = REPORT / "articulo.html"
ASSETS = REPORT / "assets"
WEB = ASSETS / "web"

# token -> (source PNG in assets/, compressed JPEG in assets/web/)
FIGS = {
    "__IMG_DATASET__": "dataset_overview",
    "__IMG_SANKEY__": "sankey",
    "__IMG_COMPARISON__": "comparison",
    "__IMG_RANKING__": "ranking",
    "__IMG_EPOCHSEL__": "epochsel",
}
JPEG_QUALITY = 88
MAX_W = 1600


def ensure_jpg(stem: str) -> Path:
    """Compress assets/<stem>.png -> assets/web/<stem>.jpg (rebuild if stale)."""
    png = ASSETS / f"{stem}.png"
    jpg = WEB / f"{stem}.jpg"
    if png.exists() and (not jpg.exists() or png.stat().st_mtime > jpg.stat().st_mtime):
        WEB.mkdir(parents=True, exist_ok=True)
        im = Image.open(png).convert("RGB")
        if im.width > MAX_W:
            h = round(im.height * MAX_W / im.width)
            im = im.resize((MAX_W, h), Image.LANCZOS)
        im.save(jpg, "JPEG", quality=JPEG_QUALITY, optimize=True)
        print(f"  compressed {stem}.png -> web/{stem}.jpg ({jpg.stat().st_size // 1024} KB)")
    return jpg


def data_uri(jpg: Path) -> str:
    b64 = base64.b64encode(jpg.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def main() -> None:
    html = SRC.read_text(encoding="utf-8")

    for token, stem in FIGS.items():
        if token not in html:
            print(f"  WARNING: token {token} not found in source")
            continue
        jpg = ensure_jpg(stem)
        html = html.replace(token, data_uri(jpg))

    remaining = [t for t in FIGS if t in html]
    assert not remaining, f"unreplaced tokens: {remaining}"

    # entity-encode all non-ASCII (base64 payload is ASCII, so it is untouched)
    html = "".join(ch if ord(ch) < 128 else f"&#{ord(ch)};" for ch in html)

    OUT.write_text(html, encoding="ascii")
    n_fig = html.count("<figure")
    n_tab = html.count("<table")
    print(f"built -> {OUT}  ({OUT.stat().st_size // 1024} KB, "
          f"{n_fig} figuras, {n_tab} tablas, {sum(ord(c) > 127 for c in html)} no-ASCII)")


if __name__ == "__main__":
    main()
