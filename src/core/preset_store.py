"""Validated storage for user-created transcription presets."""

import os
from pathlib import Path
import tempfile
from typing import Annotated

from pydantic import Field, model_validator
from src.model.options import TranscribeOptions, ValidatedModel
from src.model.presets import ParameterPreset

PresetName = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_-]{0,39}$")]


class CustomPresets(ValidatedModel):
    presets: dict[PresetName, TranscribeOptions] = Field(default_factory=dict)

    @model_validator(mode="after")
    def protect_builtins(self):
        """Prevent saved presets from replacing the built-in preset names."""
        if any(name in self.presets for name in ParameterPreset):
            raise ValueError("default and sensitive are reserved preset names")
        return self


def preset_path() -> Path:
    """Locate the per-user preset file using the XDG config directory when set."""
    root = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return root / "trscribe" / "presets.json"


def load_presets() -> dict[str, TranscribeOptions]:
    """Load validated custom presets, returning an empty collection on first use."""
    path = preset_path()
    if not path.exists():
        return {}
    return CustomPresets.model_validate_json(path.read_text(encoding="utf-8")).presets


def save_presets(presets: dict[str, TranscribeOptions]):
    """Atomically save validated presets without replacing a malformed existing file."""
    validated = CustomPresets(presets=presets)
    path = preset_path()
    existing = load_presets()
    combined = CustomPresets(presets={**existing, **validated.presets})
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None

    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(combined.model_dump_json(indent=2) + "\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
