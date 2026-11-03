"""Validated transcription presets matching the examples in test_only.py."""
from enum import Enum
from src.model.options import TranscribeOptions


class ParameterPreset(str, Enum):
    DEFAULT = 'default'
    SENSITIVE = 'sensitive'


def preset_options(preset: ParameterPreset) -> TranscribeOptions:
    """Return fresh validated settings for the selected transcription preset."""
    return TranscribeOptions(
        beam_size=10 if preset == ParameterPreset.DEFAULT else 5,
        condition_on_previous_text=False,
        vad_filter=False,
        no_speech_threshold=0.35 if preset == ParameterPreset.DEFAULT else 0.2,
        temperature=0.0,
        word_timestamps=True,
    )
