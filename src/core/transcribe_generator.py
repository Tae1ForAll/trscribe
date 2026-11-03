"""Validated faster-whisper transcription with CPU/CUDA selection."""

from pathlib import Path
from src.core.audio import prepared_audio
from src.core.device import select_device
from src.core.languages import Lang
from src.core.progress import TranscriptionProgress, report_stage
from src.model.options import (
    DeviceType,
    ModelSize,
    SubtitleSegment,
    TranscribeOptions,
    TranscriptionRequest,
)


def arrange_segments(segments, extra_display_time: float):
    """Copy and validate cues, extending their display time up to the next cue start."""
    from pydantic import TypeAdapter, Field
    from typing import Annotated

    delay = TypeAdapter(Annotated[float, Field(ge=0, allow_inf_nan=False)]).validate_python(
        extra_display_time, strict=True
    )
    validated = [SubtitleSegment.model_validate(segment, from_attributes=True).model_copy() for segment in segments]
    for i, segment in enumerate(validated):
        if i and segment.start < validated[i - 1].start:
            raise ValueError("Subtitle segments must be ordered by start time")
        available = max(0.0, validated[i + 1].start - segment.end) if i + 1 < len(validated) else delay
        segment.end += min(available, delay)
    return validated


def generate_subtitles(
    file_input: str,
    model_size: ModelSize,
    device_type: DeviceType | None = None,
    lang: Lang | None = None,
    extra_display_time: float = 1.0,
    *,
    compute_type: str = "auto",
    device_index: int = 0,
    normalize: bool = False,
    **transcribe_kwargs,
) -> list[SubtitleSegment]:
    """Validate input, run faster-whisper on a supported device, and return subtitle cues."""
    request = TranscriptionRequest(
        file_input=Path(file_input),
        model_size=model_size,
        device_type=device_type or DeviceType.CPU,
        lang=lang,
        extra_display_time=extra_display_time,
        compute_type=compute_type,
        device_index=device_index,
        normalize=normalize,
        options=TranscribeOptions(**transcribe_kwargs),
    )
    device, precision = select_device(request.device_type, request.compute_type, request.device_index)
    from faster_whisper import WhisperModel

    if request.normalize:
        report_stage("Normalizing audio...")
    with prepared_audio(request) as (audio_path, offset):
        report_stage(f"Loading model {request.model_size.value} on {device} ({precision})...")
        model = WhisperModel(
            request.model_size.value,
            device=device,
            compute_type=precision,
            device_index=request.device_index if device == "cuda" else 0,
        )

        report_stage("Decoding audio and detecting language...")
        segments, info = model.transcribe(
            str(audio_path), language=request.lang.value if request.lang else None, **request.options.model_dump()
        )

        # consume lazy inference before the temporary audio is removed.
        report_stage("Transcribing audio...")
        cues = []
        with TranscriptionProgress(getattr(info, "duration", None)) as progress:
            for segment in segments:
                cue = SubtitleSegment.model_validate(segment, from_attributes=True)
                cues.append(cue)
                progress.update(cue.end)
        validated = arrange_segments(cues, 0.0)

        if offset == 0.0:
            return arrange_segments(validated, request.extra_display_time)
        aligned = [
            SubtitleSegment(start=max(0.0, cue.start + offset), end=max(0.0, cue.end + offset), text=cue.text)
            for cue in validated
            if cue.end + offset > 0.0
        ]

        return arrange_segments(aligned, request.extra_display_time)
