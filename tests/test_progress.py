"""Progress rendering and lazy transcription integration tests."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pydantic import ValidationError
from src.core.progress import TranscriptionProgress
from src.core.transcribe_generator import generate_subtitles
from src.model.options import ModelSize


class TerminalBuffer(io.StringIO):
    def isatty(self):
        """Emulate a terminal so live progress can be checked without a real console."""
        return True


class ProgressTests(unittest.TestCase):
    def test_completion_includes_trailing_silence(self):
        """Reach full duration on success even when the last cue ends before silence."""
        output = TerminalBuffer()
        with contextlib.redirect_stderr(output), TranscriptionProgress(10.0) as progress:
            progress.update(4.0)
            progress.render(force=True)
            self.assertIn('40%', output.getvalue())
        self.assertEqual(progress.timing.processed, 10.0)
        self.assertIn('100%', output.getvalue())
        self.assertTrue(output.getvalue().endswith('\n'))

    def test_progress_is_monotonic_bounded_and_validated(self):
        """Clamp overshoots, ignore backwards timestamps, and reject invalid timing."""
        with contextlib.redirect_stderr(TerminalBuffer()), TranscriptionProgress(10.0) as progress:
            progress.update(5.0)
            progress.update(3.0)
            self.assertEqual(progress.timing.processed, 5.0)
            progress.update(12.0)
            self.assertEqual(progress.timing.processed, 10.0)
            with self.assertRaises(ValidationError):
                progress.update(float('nan'))
        with self.assertRaises(ValidationError):
            TranscriptionProgress(-1.0)

    def test_failure_and_interrupt_do_not_report_completion(self):
        """Close the display without claiming 100 percent on inference failure or Ctrl-C."""
        for error in (RuntimeError('failed inference'), KeyboardInterrupt()):
            output = TerminalBuffer()
            with contextlib.redirect_stderr(output), self.assertRaises(type(error)):
                with TranscriptionProgress(10.0) as progress:
                    progress.update(10.0)
                    raise error
            self.assertNotIn('100%', output.getvalue())
            self.assertTrue(output.getvalue().endswith('\n'))
            self.assertIn('interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', output.getvalue())

    def test_unknown_or_zero_duration(self):
        """Display processed seconds without inventing a percentage when duration is absent."""
        for duration in (None, 0.0):
            output = TerminalBuffer()
            with contextlib.redirect_stderr(output), TranscriptionProgress(duration) as progress:
                progress.update(3.0)
            self.assertIn('3.0s processed', output.getvalue())
            self.assertNotIn('%', output.getvalue())

    def test_redirected_output_has_no_animation(self):
        """Emit a plain completion message instead of carriage returns when output is piped."""
        output = io.StringIO()
        with contextlib.redirect_stderr(output), TranscriptionProgress(10.0) as progress:
            progress.update(3.0)
        self.assertEqual(output.getvalue(), 'Transcription complete.\n')

    def test_transcription_updates_from_raw_timestamps(self):
        """Report lazy segment progress before subtitle display-time extensions are applied."""
        with tempfile.TemporaryDirectory() as directory:
            media = Path(directory) / 'sample.wav'
            media.write_bytes(b'fixture')
            segments = [SimpleNamespace(start=0.0, end=2.0, text='first'),
                        SimpleNamespace(start=4.0, end=6.0, text='second')]
            model = SimpleNamespace(transcribe=lambda *a, **k: (iter(segments), SimpleNamespace(duration=10.0)))
            with patch('src.core.transcribe_generator.select_device', return_value=('cpu', 'int8')), \
                 patch.dict(sys.modules, {'faster_whisper': SimpleNamespace(WhisperModel=lambda *a, **k: model)}), \
                 patch('src.core.transcribe_generator.TranscriptionProgress') as progress, \
                 contextlib.redirect_stderr(io.StringIO()):
                cues = generate_subtitles(str(media), ModelSize.TINY, extra_display_time=1.0)
            progress.assert_called_once_with(10.0)
            updates = progress.return_value.__enter__.return_value.update.call_args_list
            self.assertEqual([call.args[0] for call in updates], [2.0, 6.0])
            self.assertEqual([cue.end for cue in cues], [3.0, 7.0])


if __name__ == '__main__':
    unittest.main()
