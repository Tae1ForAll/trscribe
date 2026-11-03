"""Optional audio preparation adapted from the root repaire.py utility."""

from contextlib import contextmanager
from pathlib import Path
import shutil
import subprocess
import tempfile

from pydantic import ConfigDict, Field, field_validator
from src.model.options import TranscriptionRequest, ValidatedModel


class AudioTiming(ValidatedModel):
    model_config = ConfigDict(extra="ignore")
    start_time: float | None = None
    duration: float | None = Field(default=None, gt=0)

    @field_validator("start_time", "duration", mode="before")
    @classmethod
    def parse_seconds(cls, value):
        """Parse ffprobe decimal strings while allowing absent timing metadata."""
        if value is None or value == "N/A":
            return None
        return float(value) if isinstance(value, str) else value


class ProbeResult(ValidatedModel):
    # FFprobe versions may also emit unrelated programs or stream-group sections.
    model_config = ConfigDict(extra="ignore")
    streams: list[AudioTiming] = Field(min_length=1)
    format: AudioTiming = Field(default_factory=AudioTiming)


class TempoRequest(ValidatedModel):
    speed: float = Field(gt=0)


def run_media_tool(command):
    """Run a media tool without a shell and turn failures into actionable errors."""
    try:
        return subprocess.run(command, capture_output=True, text=True, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Normalization requires {command[0]} on PATH.") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"{command[0]} failed during normalization: {exc.stderr.strip()}") from exc


def probe_audio(path: Path) -> ProbeResult:
    """Read and validate timing metadata for the first audio stream and its container."""
    request = TranscriptionRequest(file_input=path)
    result = run_media_tool(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=start_time,duration:format=start_time",
            "-of",
            "json",
            str(request.file_input.resolve()),
        ]
    )
    return ProbeResult.model_validate_json(result.stdout)


def tempo_filter(speed: float) -> str:
    """Build tempo filters with individual factors between 0.5 and 2.0."""
    speed = TempoRequest(speed=speed).speed
    factors = []
    while speed > 2.0:
        factors.append(2.0)
        speed /= 2.0
    while speed < 0.5:
        factors.append(0.5)
        speed /= 0.5
    factors.append(speed)
    return ",".join(f"atempo={factor:.17g}" for factor in factors)


@contextmanager
def prepared_audio(request: TranscriptionRequest):
    """Yield an input path and timeline offset, cleaning normalized audio after use."""
    request = TranscriptionRequest.model_validate(request)
    if not request.normalize:
        yield request.file_input, 0.0
        return
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise RuntimeError(f"--normalize requires {tool} on PATH. Install FFmpeg or omit --normalize.")

    original = probe_audio(request.file_input)
    timing = original.streams[0]
    # Use playback-relative time, rather than a container's absolute timestamp.
    offset = (timing.start_time or 0.0) - (original.format.start_time or 0.0)
    with tempfile.TemporaryDirectory(prefix="transcriber-") as directory:
        normalized = Path(directory) / "normalized.wav"
        run_media_tool(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-y",
                "-i",
                str(request.file_input.resolve()),
                "-map",
                "0:a:0",
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(normalized),
            ]
        )
        repaired = probe_audio(normalized).streams[0]
        if repaired.duration is None:
            raise ValueError("Cannot determine normalized audio duration")
        if timing.duration is not None and abs(repaired.duration - timing.duration) > 0.01:
            corrected = Path(directory) / "corrected.wav"
            run_media_tool(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-v",
                    "error",
                    "-y",
                    "-i",
                    str(normalized),
                    "-af",
                    tempo_filter(repaired.duration / timing.duration),
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-c:a",
                    "pcm_s16le",
                    str(corrected),
                ]
            )
            normalized = corrected
        yield normalized, offset
