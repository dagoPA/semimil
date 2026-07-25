#!/usr/bin/env python
"""Extract the MIIC dataset from ``dataverse_files.zip``.

Reproducible & argument-free: all paths come from ``configs/data.yaml``.

The outer zip contains four RAR5 bundles whose entries use the "store" method
(the images are already-compressed PNGs), so ``rarfile`` reads them natively in
pure Python -- no unrar/7z/unar binary is required.

Run:  uv run python scripts/extract_data.py
"""

from __future__ import annotations

import shutil
import subprocess
import time
import zipfile
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "data.yaml"


def load_config() -> dict:
    with CONFIG_PATH.open() as fh:
        return yaml.safe_load(fh)


def unzip_bundles(archive: Path, raw_dir: Path, bundles: list[str]) -> None:
    """Extract the RAR bundles (and text metadata) from the outer zip."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        members = set(zf.namelist())
        wanted = list(bundles) + [m for m in ("readme.txt", "MANIFEST.TXT") if m in members]
        for name in wanted:
            out = raw_dir / name
            if out.exists() and out.stat().st_size > 0:
                continue
            print(f"  unzip {name}")
            zf.extract(name, raw_dir)


def extract_rar(rar_path: Path, raw_dir: Path) -> int:
    """Extract a whole RAR5 bundle in one ``unrar`` call.

    The bundles are mostly compressed (method 0x33), so a native reader is not
    enough; ``unrar`` (rarlab static binary on PATH) decompresses them. One call
    per bundle avoids spawning a process per file. Returns the file count under
    the bundle's top-level directory.
    """
    unrar = shutil.which("unrar")
    if unrar is None:
        raise RuntimeError(
            "unrar not found on PATH. Install the rarlab static binary:\n"
            "  curl -sSL https://www.rarlab.com/rar/rarlinux-x64-711.tar.gz | tar xz\n"
            "  cp rar/unrar ~/.local/bin/"
        )
    stem = raw_dir / rar_path.stem
    already = stem.is_dir() and any(stem.rglob("*"))
    if not already:
        # -o+ overwrite, -idq quiet; trailing slash = destination directory.
        subprocess.run(
            [unrar, "x", "-o+", "-idq", str(rar_path), f"{raw_dir}/"],
            check=True,
        )
    return sum(1 for _ in stem.rglob("*") if _.is_file())


def main() -> None:
    cfg = load_config()
    archive = Path(cfg["data_root"]) / cfg["archive_name"]
    raw_dir = Path(cfg["raw_dir"])
    bundles = cfg["bundles"]

    print(f"archive : {archive}")
    print(f"raw_dir : {raw_dir}\n")

    print("[1/2] unzip outer archive ...")
    unzip_bundles(archive, raw_dir, bundles)

    print("[2/2] extract RAR bundles (native store read) ...")
    for bundle in bundles:
        rar_path = raw_dir / bundle
        t0 = time.time()
        n = extract_rar(rar_path, raw_dir)
        print(f"  {bundle:<24} {n:>6} files  ({time.time() - t0:5.1f}s)")
        if not cfg.get("keep_rar", True):
            rar_path.unlink(missing_ok=True)

    print("\ndone.")


if __name__ == "__main__":
    main()
