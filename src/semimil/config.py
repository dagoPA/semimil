"""Central, typed configuration.

All paths default to the shared project space on ``crater_001`` but can be
overridden with ``SEMIMIL_*`` environment variables (or a local ``.env`` file),
which keeps experiments reproducible and portable across machines.

Example
-------
>>> from semimil import settings
>>> settings.raw_dir            # extracted MIIC dataset
>>> settings.outputs_dir        # runs / checkpoints / logs
"""

from __future__ import annotations

from pathlib import Path

from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

#: Shared project space where the raw dataset and all outputs live.
DEFAULT_DATA_ROOT = Path("/autofs/space/crater_001/projects/micropath/semi")


class Settings(BaseSettings):
    """Runtime configuration resolved from env vars, ``.env`` and defaults."""

    model_config = SettingsConfigDict(
        env_prefix="SEMIMIL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- storage layout ---------------------------------------------------
    data_root: Path = DEFAULT_DATA_ROOT
    #: Name of the original archive shipped by the dataset authors.
    archive_name: str = "dataverse_files.zip"

    # --- reproducibility --------------------------------------------------
    seed: int = 42
    #: Torch device string; "auto" -> cuda if available else cpu.
    device: str = "auto"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def archive_path(self) -> Path:
        """Absolute path to the original ``dataverse_files.zip``."""
        return self.data_root / self.archive_name

    @computed_field  # type: ignore[prop-decorator]
    @property
    def raw_dir(self) -> Path:
        """Directory holding the extracted MIIC dataset (the four RAR bundles)."""
        return self.data_root / "MIIC"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def outputs_dir(self) -> Path:
        """Root for run artefacts: checkpoints, logs, features, figures."""
        return self.data_root / "outputs"

    def resolved_device(self) -> str:
        """Return a concrete device string, resolving ``"auto"``."""
        if self.device != "auto":
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ModuleNotFoundError:  # pragma: no cover
            return "cpu"


#: Import-time singleton; import this rather than constructing ``Settings()``.
settings = Settings()
