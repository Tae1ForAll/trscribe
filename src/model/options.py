"""Validated boundaries for CLI, transcription, and subtitle output."""

from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from src.core.languages import Lang


class ModelSize(str, Enum):
    TINY = "tiny"
    BASE = "base"
    SMALL = "small"
    MEDIUM = "medium"
    MEDIUM_EN = "medium.en"
    LARGE = "large"
    LARGE_V3 = "large-v3"
    TURBO = "turbo"


class DeviceType(str, Enum):
    AUTO = "auto"
    NVIDIA_CUDA = "cuda"
    CPU = "cpu"


class ValidatedModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class TranscribeOptions(ValidatedModel):
    beam_size: int = Field(default=5, ge=1)
    condition_on_previous_text: bool = True
    vad_filter: bool = False
    no_speech_threshold: float = Field(default=0.6, ge=0, le=1)
    temperature: float = Field(default=0.0, ge=0, le=1)
    word_timestamps: bool = False
    task: Literal["transcribe", "translate"] = "transcribe"
    initial_prompt: str | None = None


class TranscriptionRequest(ValidatedModel):
    file_input: Path
    model_size: ModelSize = ModelSize.SMALL
    device_type: DeviceType = DeviceType.AUTO
    lang: Lang | None = None
    extra_display_time: float = Field(default=0.0, ge=0)
    compute_type: Literal["auto", "float32", "float16", "int8", "int8_float16", "int8_float32"] = "auto"
    device_index: int = Field(default=0, ge=0)
    normalize: bool = False
    options: TranscribeOptions = Field(default_factory=TranscribeOptions)

    @field_validator("file_input")
    @classmethod
    def readable_file(cls, value):
        """Require an existing, readable, nonempty input file."""
        if not value.is_file():
            raise ValueError(f"Input file does not exist or is not a file: {value}")
        with value.open("rb") as stream:
            if not stream.read(1):
                raise ValueError("Input file is empty")
        return value


class SubtitleSegment(ValidatedModel):
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    text: str

    @model_validator(mode="after")
    def ordered_times(self):
        """Reject subtitle cues whose end precedes their start."""
        if self.end < self.start:
            raise ValueError("Subtitle end must not precede its start")
        return self


class ExportRequest(ValidatedModel):
    path: Path
    format: Literal["srt", "vtt", "ass"]
    overwrite: bool = False

    @field_validator("path")
    @classmethod
    def output_parent(cls, value):
        """Require an existing output directory and reject directory-valued output paths."""
        if not value.parent.is_dir():
            raise ValueError(f"Output directory does not exist: {value.parent}")

        if value.is_dir():
            raise ValueError("Output path is a directory")
        return value
