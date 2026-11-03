"""Audio preparation and transcription integration tests without model downloads."""
import contextlib
import io
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import wave

from pydantic import ValidationError
from src.cli import main
from src.core.audio import AudioTiming, ProbeResult, prepared_audio, tempo_filter
from src.core.transcribe_generator import generate_subtitles
from src.model.options import ModelSize, TranscriptionRequest


class AudioTests(unittest.TestCase):
    def setUp(self):
        """Create a small stereo WAV fixture in an isolated temporary directory."""
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.input = Path(self.directory.name) / 'input.wav'
        with wave.open(str(self.input), 'wb') as stream:
            stream.setnchannels(2)
            stream.setsampwidth(2)
            stream.setframerate(44100)
            stream.writeframes(b'\x00\x00\x00\x00' * 44100)

    def test_disabled_normalization_needs_no_external_tools(self):
        """Verify the default path uses the original media without invoking tools."""
        with patch('src.core.audio.run_media_tool') as run:
            with prepared_audio(TranscriptionRequest(file_input=self.input)) as (path, offset):
                self.assertEqual(path, self.input)
                self.assertEqual(offset, 0.0)
            run.assert_not_called()

    def test_metadata_validation(self):
        """Reject invalid timing metadata while accepting absent duration and decimal strings."""
        self.assertEqual(AudioTiming(duration='1.25').duration, 1.25)
        self.assertIsNone(AudioTiming(duration='N/A').duration)
        for value in ('nan', 'inf', '-1', '0', True):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                AudioTiming(duration=value)
        with self.assertRaises(ValidationError):
            ProbeResult.model_validate_json('{"streams": []}')
        with self.assertRaises(ValidationError):
            TranscriptionRequest(file_input=self.input, normalize='yes')

    def test_missing_tools(self):
        """Report an actionable error when normalization tools are unavailable."""
        with patch('src.core.audio.shutil.which', return_value=None):
            with self.assertRaisesRegex(RuntimeError, 'requires ffmpeg'):
                with prepared_audio(TranscriptionRequest(file_input=self.input, normalize=True)):
                    self.fail('Normalization should not proceed')

    def test_duration_correction_and_cleanup(self):
        """Apply speed correction and preserve the relative stream start offset."""
        original = ProbeResult(streams=[AudioTiming(start_time=12.0, duration=1.0)],
                               format=AudioTiming(start_time=10.0))
        converted = ProbeResult(streams=[AudioTiming(duration=1.1)])
        with patch('src.core.audio.shutil.which', return_value='/tool'), \
             patch('src.core.audio.probe_audio', side_effect=[original, converted]), \
             patch('src.core.audio.run_media_tool') as run:
            with prepared_audio(TranscriptionRequest(file_input=self.input, normalize=True)) as (path, offset):
                self.assertEqual(path.name, 'corrected.wav')
                self.assertEqual(offset, 2.0)
                command = run.call_args.args[0]
                self.assertAlmostEqual(float(command[command.index('-af') + 1].split('=')[1]), 1.1)
                directory = path.parent
                self.assertTrue(directory.exists())
            self.assertFalse(directory.exists())

    def test_missing_duration_skips_speed_correction(self):
        """Avoid guessing an audio duration from a longer video container."""
        with patch('src.core.audio.shutil.which', return_value='/tool'), \
             patch('src.core.audio.probe_audio', side_effect=[ProbeResult(streams=[AudioTiming()]),
                                                            ProbeResult(streams=[AudioTiming(duration=1.0)])]), \
             patch('src.core.audio.run_media_tool') as run:
            with prepared_audio(TranscriptionRequest(file_input=self.input, normalize=True)) as (path, _):
                self.assertEqual(path.name, 'normalized.wav')
            self.assertEqual(run.call_count, 1)

    def test_tempo_factor_chaining(self):
        """Keep each FFmpeg tempo factor in the quality-preserving 0.5 to 2 range."""
        for speed in (0.1, 1.0, 1.1, 8.0):
            factors = [float(part.split('=')[1]) for part in tempo_filter(speed).split(',')]
            product = 1.0
            for factor in factors:
                self.assertTrue(0.5 <= factor <= 2.0)
                product *= factor
            self.assertAlmostEqual(product, speed)

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg tools are not installed')
    def test_real_normalization_and_cleanup_on_error(self):
        """Verify real conversion produces mono 16 kHz PCM and removes files even on failure."""
        original = self.input.read_bytes()
        with self.assertRaisesRegex(RuntimeError, 'inference failed'):
            with prepared_audio(TranscriptionRequest(file_input=self.input, normalize=True)) as (path, offset):
                self.assertEqual(offset, 0.0)
                with wave.open(str(path), 'rb') as stream:
                    self.assertEqual((stream.getnchannels(), stream.getframerate(), stream.getsampwidth()), (1, 16000, 2))
                    self.assertAlmostEqual(stream.getnframes() / stream.getframerate(), 1.0, places=2)
                raise RuntimeError('inference failed')
        self.assertFalse(path.exists())
        self.assertEqual(self.input.read_bytes(), original)

    def test_transcription_consumes_normalized_audio_and_restores_offset(self):
        """Keep prepared audio alive through lazy inference and align cues to the source."""
        normalized = Path(self.directory.name) / 'normalized.wav'

        @contextlib.contextmanager
        def prepare(request):
            """Emulate a normalized file that only exists inside the preparation context."""
            self.assertTrue(request.normalize)
            normalized.write_bytes(b'audio')
            try:
                yield normalized, 2.0
            finally:
                normalized.unlink()

        def transcribe(path, **kwargs):
            """Return a lazy cue stream that requires the normalized file to remain present."""
            self.assertEqual(Path(path), normalized)

            def cues():
                """Check the file lifetime during lazy transcription."""
                self.assertTrue(normalized.exists())
                yield SimpleNamespace(start=0.0, end=1.0, text='hello')

            return cues(), None

        with patch('src.core.transcribe_generator.prepared_audio', side_effect=prepare), \
             patch('src.core.transcribe_generator.select_device', return_value=('cpu', 'int8')), \
             patch.dict(sys.modules, {'faster_whisper': SimpleNamespace(WhisperModel=lambda *a, **k: SimpleNamespace(transcribe=transcribe))}):
            cues = generate_subtitles(str(self.input), ModelSize.TINY, normalize=True, extra_display_time=0.0)
        self.assertEqual((cues[0].start, cues[0].end), (2.0, 3.0))
        self.assertFalse(normalized.exists())

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg tools are not installed')
    def test_real_video_with_delayed_audio(self):
        """Preserve a delayed audio stream's offset relative to video playback."""
        video = Path(self.directory.name) / 'delayed.mkv'
        subprocess.run([
            'ffmpeg', '-nostdin', '-v', 'error', '-y', '-f', 'lavfi',
            '-i', 'color=size=16x16:rate=1:duration=4', '-itsoffset', '2',
            '-i', str(self.input), '-map', '0:v:0', '-map', '1:a:0',
            '-c:v', 'mpeg4', '-c:a', 'pcm_s16le', str(video),
        ], check=True, capture_output=True)
        with prepared_audio(TranscriptionRequest(file_input=video, normalize=True)) as (path, offset):
            self.assertAlmostEqual(offset, 2.0, places=2)
            self.assertTrue(path.is_file())
        self.assertFalse(path.exists())

    def test_conversion_failure_removes_temporary_directory(self):
        """Remove partially written audio if FFmpeg conversion fails."""
        paths = []

        def fail_conversion(command):
            """Create a partial output file before simulating a conversion failure."""
            path = Path(command[-1])
            path.write_bytes(b'partial')
            paths.append(path)
            raise RuntimeError('conversion failed')

        with patch('src.core.audio.shutil.which', return_value='/tool'), \
             patch('src.core.audio.probe_audio', return_value=ProbeResult(streams=[AudioTiming(duration=1.0)])), \
             patch('src.core.audio.run_media_tool', side_effect=fail_conversion):
            with self.assertRaisesRegex(RuntimeError, 'conversion failed'):
                with prepared_audio(TranscriptionRequest(file_input=self.input, normalize=True)):
                    self.fail('Normalization should fail')
        self.assertFalse(paths[0].parent.exists())

    def test_cli_normalization_choice(self):
        """Forward explicit opt-in and opt-out choices to the transcription pipeline."""
        for flag, expected in [('--normalize', True), ('--no-normalize', False)]:
            with patch('src.core.transcribe_generator.generate_subtitles', return_value=[]) as generate, \
                 patch('src.core.file_manager.FileManager.export_subtitles'), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(['transcribe', str(self.input), flag]), 0)
                self.assertEqual(generate.call_args.kwargs['normalize'], expected)

    def test_tool_failure_is_reported(self):
        """Convert FFmpeg failures into errors the CLI can display without a traceback."""
        from src.core.audio import run_media_tool
        with patch('src.core.audio.subprocess.run', side_effect=subprocess.CalledProcessError(1, ['ffmpeg'], stderr='invalid audio')):
            with self.assertRaisesRegex(RuntimeError, 'invalid audio'):
                run_media_tool(['ffmpeg'])


if __name__ == '__main__':
    unittest.main()
