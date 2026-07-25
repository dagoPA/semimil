"""Extract the raw MIIC dataset.

The dataset ships as ``dataverse_files.zip`` which itself contains four RAR
bundles (``Anomaly_train``, ``Anomaly_test``, ``Inpainting_train``,
``Inpainting_test``) plus text metadata. Extracting RAR needs a backend binary
on ``PATH`` (``unrar``, ``unar`` or ``7z``); install one, e.g. via the shared
micromamba::

    micromamba create -y -n tools -c conda-forge unar
    micromamba run -n tools unar -o <dest> <file>.rar

This module first unzips the outer archive, then extracts each RAR through the
``rarfile`` backend (``pip install .[data]``) if one is available.
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

from semimil.config import settings

RAR_BUNDLES = (
    "Anomaly_train.rar",
    "Anomaly_test.rar",
    "Inpainting_train.rar",
    "Inpainting_test.rar",
)


def _find_rar_backend() -> str | None:
    """Return the first available RAR-capable binary, or ``None``."""
    for tool in ("unar", "unrar", "7z", "7za", "bsdtar"):
        if shutil.which(tool):
            return tool
    return None


def unzip_outer(archive_path: Path, dest: Path) -> list[Path]:
    """Unzip the outer ``dataverse_files.zip`` into *dest*.

    Returns the paths of the extracted members.
    """
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as zf:
        zf.extractall(dest)
        return [dest / name for name in zf.namelist()]


def extract_rar(rar_path: Path, dest: Path) -> None:
    """Extract a single RAR bundle into *dest*.

    Raises
    ------
    RuntimeError
        If no RAR backend binary is available on ``PATH``.
    """
    import rarfile  # local import: only needed for extraction

    backend = _find_rar_backend()
    if backend is None:
        raise RuntimeError(
            "No RAR backend found on PATH. Install one (unar / unrar / 7z), e.g.:\n"
            "  micromamba create -y -n tools -c conda-forge unar\n"
            "then re-run with that environment active."
        )
    dest.mkdir(parents=True, exist_ok=True)
    with rarfile.RarFile(rar_path) as rf:
        rf.extractall(dest)


def prepare_miic(
    archive_path: Path | None = None,
    raw_dir: Path | None = None,
    *,
    bundles: tuple[str, ...] = RAR_BUNDLES,
) -> Path:
    """End-to-end extraction of the MIIC dataset.

    Unzips the outer archive into *raw_dir*, then extracts each RAR bundle in
    place. Idempotent: bundles whose target directory already exists are skipped.

    Returns the ``raw_dir`` containing the extracted dataset.
    """
    archive_path = archive_path or settings.archive_path
    raw_dir = raw_dir or settings.raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)

    unzip_outer(archive_path, raw_dir)

    for bundle in bundles:
        rar_path = raw_dir / bundle
        target = raw_dir / Path(bundle).stem
        if target.exists():
            continue
        extract_rar(rar_path, raw_dir)

    return raw_dir
