"""Dependency-free terminal progress reporting for media transcription."""

import shutil
import sys
import time
import threading

from pydantic import Field
from src.model.options import ValidatedModel


class ProgressTiming(ValidatedModel):
    duration: float | None = Field(default=None, ge=0)
    processed: float = Field(default=0.0, ge=0)


class LoadingIndicator:
    def __init__(self, message: str, stream=None):
        """Prepare an animated terminal indicator with a plain-text fallback."""
        self.message = message
        self.stream = stream if stream is not None else sys.stderr
        self.live = self.stream.isatty()
        self.stop = threading.Event()
        self.thread = None
        self.last_width = 0

    def __enter__(self):
        """Display feedback immediately and animate it while the operation runs."""
        if self.live:
            self.render('|')
            self.thread = threading.Thread(target=self.animate, daemon=True)
            self.thread.start()
        else:
            print(self.message, file=self.stream, flush=True)
        return self

    def render(self, frame: str):
        """Draw a spinner frame within the current terminal width."""
        width = max(1, shutil.get_terminal_size(fallback=(80, 24)).columns - 1)
        label = f'  {frame} {self.message}'[:width]
        padding = max(0, min(self.last_width, width) - len(label))
        self.stream.write('\r' + label + ' ' * padding)
        self.stream.flush()
        self.last_width = len(label)

    def animate(self):
        """Cycle spinner frames until context exit signals completion."""
        frames = '/-\\|'
        index = 0
        while not self.stop.wait(0.1):
            self.render(frames[index % len(frames)])
            index += 1

    def __exit__(self, exc_type, exc_value, traceback):
        """Stop animation and clear its line on success, failure, or interruption."""
        self.stop.set()
        if self.thread is not None:
            self.thread.join()
        if self.live:
            self.stream.write('\r' + ' ' * self.last_width + '\r')
            self.stream.flush()


def report_stage(message: str):
    """Print a processing stage to stderr without mixing it with command output."""
    print(message, file=sys.stderr, flush=True)


def elapsed_text(seconds: float) -> str:
    """Format elapsed seconds as minutes and seconds for the progress display."""
    minutes, seconds = divmod(int(seconds), 60)
    return f"{minutes:02}:{seconds:02}"


class TranscriptionProgress:
    def __init__(self, duration: float | None):
        """Initialize validated timing and enable live rendering only for terminals."""
        self.timing = ProgressTiming(duration=duration)
        self.stream = sys.stderr
        self.live = self.stream.isatty()
        self.started = time.monotonic()
        self.last_render = float("-inf")
        self.last_width = 0

    def __enter__(self):
        """Display the initial transcription progress indicator."""
        self.render(force=True)
        return self

    def update(self, processed: float):
        """Advance progress monotonically, bounded by the reported media duration."""
        value = ProgressTiming(processed=processed).processed
        value = max(self.timing.processed, value)
        if self.timing.duration:
            value = min(value, self.timing.duration)
        self.timing.processed = value
        self.render()

    def render(self, *, force=False, status=None):
        """Render a compact percentage bar or processed time when duration is unknown."""
        if not self.live:
            return
        now = time.monotonic()
        if not force and now - self.last_render < 0.1:
            return
        self.last_render = now
        elapsed = elapsed_text(now - self.started)
        duration = self.timing.duration
        heading = f"Transcription {status}" if status else "Transcribing"

        if duration:
            ratio = min(self.timing.processed / duration, 1.0)

            # Reserve 100% for successful exhaustion of the lazy inference iterator.
            percentage = 100 if status == "done" else min(int(ratio * 100), 99)
            bar = "#" * (percentage // 5) + "-" * (20 - percentage // 5)
            label = f"{heading} [{bar}] {percentage:3d}%"
            label += f" | {self.timing.processed:.1f}/{duration:.1f}s | elapsed {elapsed}"

        else:
            label = f"{heading} | {self.timing.processed:.1f}s processed | elapsed {elapsed}"

        width = max(1, shutil.get_terminal_size(fallback=(80, 24)).columns - 1)
        label = label[:width]
        padding = max(0, min(self.last_width, width) - len(label))
        self.stream.write("\r" + label + " " * padding)
        self.stream.flush()
        self.last_width = len(label)

    def __exit__(self, exc_type, exc_value, traceback):
        """Finish the line and distinguish completion from failed or interrupted inference."""
        if exc_type is None:
            if self.timing.duration:
                self.timing.processed = self.timing.duration
            status = "done"
        else:
            status = "interrupted" if issubclass(exc_type, KeyboardInterrupt) else "failed"
        self.render(force=True, status=status)
        if self.live:
            self.stream.write("\n")
            self.stream.flush()
        elif exc_type is None:
            report_stage("Transcription complete.")
