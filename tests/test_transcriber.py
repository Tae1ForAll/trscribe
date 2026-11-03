import contextlib
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pydantic import ValidationError
from src.cli import main
from src.core.device import check_device, select_device, DeviceReport
from src.core.file_manager import FileManager, FileType, timestamp
from src.core.transcribe_generator import arrange_segments, generate_subtitles
from src.model.options import DeviceType, ModelSize, SubtitleSegment, TranscriptionRequest, TranscribeOptions


class TranscriberTests(unittest.TestCase):
    def test_installed_entrypoint_routes_commands(self):
        """Verify the launcher prepares CUDA and handles direct paths, subcommands, and help."""
        from src.cli import entrypoint

        cases = [
            (["recording.mp4", "--device", "cuda"], ["transcribe", "recording.mp4", "--device", "cuda"]),
            (["check-device"], ["check-device"]),
            (["transcribe", "recording.mp4"], ["transcribe", "recording.mp4"]),
            (["--help"], ["--help"]),
        ]
        for arguments, expected in cases:
            with self.subTest(arguments=arguments), \
                 patch('src.core.environment.setup') as setup, \
                 patch('src.cli.sys.argv', ['transcribe', *arguments]), \
                 patch('src.cli.main', return_value=0) as main:
                self.assertEqual(entrypoint(), 0)
                setup.assert_called_once_with()
                main.assert_called_once_with(expected)

    def test_cuda_library_setup_preserves_module_invocation(self):
        """Verify CUDA setup restarts module commands once with installed library paths."""
        from src.core.environment import setup
        with patch('src.core.environment.sys.platform', 'linux'), \
             patch('src.core.environment.site.getsitepackages', return_value=['/packages']), \
             patch('src.core.environment.site.getusersitepackages', return_value='/packages'), \
             patch('src.core.environment.Path.is_dir', return_value=True), \
             patch('src.core.environment.sys.orig_argv', ['python', '-m', 'src', 'check-device']), \
             patch.dict('os.environ', {'LD_LIBRARY_PATH': '/existing'}, clear=True), \
             patch('src.core.environment.os.execve') as restart:
            setup()
            executable, arguments, environment = restart.call_args.args
            self.assertEqual(arguments, [executable, '-m', 'src', 'check-device'])
            self.assertEqual(environment['LD_LIBRARY_PATH'],
                             '/packages/nvidia/cublas/lib:/packages/nvidia/cudnn/lib:/existing')
            restart.reset_mock()
            with patch.dict('os.environ', environment, clear=True):
                setup()
            restart.assert_not_called()

    def setUp(self):
        """Create temporary input media and sample subtitle cues for each test."""
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.media = Path(self.temp.name) / 'audio.wav'
        self.media.write_bytes(b'placeholder')
        self.segments = [SubtitleSegment(start=0.0, end=1.25, text='Hello <world>'),
                         SubtitleSegment(start=2.0, end=3.0, text='こんにちは')]

    def test_input_validation(self):
        """Verify invalid paths, timings, indices, and transcription options are rejected."""
        for kwargs in [{'file_input': self.media, 'extra_display_time': -1.0},
                       {'file_input': self.media, 'extra_display_time': float('nan')},
                       {'file_input': self.media, 'device_index': True},
                       {'file_input': self.media.parent}]:
            with self.assertRaises(ValidationError):
                TranscriptionRequest(**kwargs)
        with self.assertRaises(ValidationError):
            TranscribeOptions(beam_size='5')
        with self.assertRaises(ValidationError):
            TranscribeOptions(unknown=True)
        with self.assertRaises(ValidationError):
            SubtitleSegment(start=2.0, end=1.0, text='invalid')

    def test_delay_capped_and_does_not_mutate_input(self):
        """Verify cue extensions stop at the next cue and leave original cues unchanged."""
        result = arrange_segments(self.segments, 2.0)
        self.assertEqual([s.end for s in result], [2.0, 5.0])
        self.assertEqual(self.segments[0].end, 1.25)

    def test_exports_and_overwrite_protection(self):
        """Verify subtitle formats, Unicode output, SRT import, and overwrite protection."""
        for format in FileType:
            target = Path(self.temp.name) / ('sub.' + format.value)
            FileManager.export_subtitles(self.segments, format, target)
            content = target.read_text()
            self.assertIn('こんにちは', content)
            if format == FileType.VTT:
                self.assertTrue(content.startswith('WEBVTT\n\n'))
                self.assertIn('00:00:01.250', content)
                self.assertIn('&lt;world&gt;', content)
            if format == FileType.ASS:
                self.assertIn('[Events]', content)
                self.assertIn('00:00:01.25', content)
            with self.assertRaises(FileExistsError):
                FileManager.export_subtitles(self.segments, format, target)
        imported = FileManager.import_subtitles(Path(self.temp.name) / 'sub.srt')
        self.assertEqual(imported[0].end.total_seconds(), 1.25)

    def test_timestamp_carry(self):
        """Verify timestamp rounding carries into minutes and supports long durations."""
        self.assertEqual(timestamp(59.9999), '00:01:00.000')
        self.assertEqual(timestamp(360000.0), '100:00:00.000')

    def test_gpu_preflight(self):
        """Verify GPU detection handles available devices, invalid indices, and missing libraries."""
        backend = SimpleNamespace(get_cuda_device_count=lambda: 1,
                                  get_supported_compute_types=lambda *args: {'float32', 'float16'})
        with patch.dict(sys.modules, {'ctranslate2': backend}), patch('src.core.device.ctypes.CDLL'):
            self.assertTrue(check_device().cuda_available)
            self.assertFalse(check_device(1).cuda_available)
        with patch.dict(sys.modules, {'ctranslate2': backend}), patch('src.core.device.ctypes.CDLL', side_effect=OSError('missing')):
            self.assertFalse(check_device().cuda_available)
        with self.assertRaises(ValidationError):
            check_device(-1)

    def test_cpu_fallback_and_explicit_cuda_failure(self):
        """Verify automatic CPU fallback and errors for unsupported device or precision choices."""
        report = DeviceReport(cpu_compute_types=['int8', 'float32'], issues=['No GPU'])
        with patch('src.core.device.check_device', return_value=report):
            self.assertEqual(select_device(DeviceType.AUTO, 'auto'), ('cpu', 'int8'))
            with self.assertRaises(RuntimeError):
                select_device(DeviceType.NVIDIA_CUDA, 'auto')
            with self.assertRaises(ValueError):
                select_device(DeviceType.CPU, 'float16')

    def test_transcription_passes_cpu_precision_and_consumes_generator(self):
        """Verify mocked transcription consumes lazy segments and returns validated cues."""
        model = SimpleNamespace(transcribe=lambda *args, **kwargs: (iter(self.segments), None))
        with patch('src.core.transcribe_generator.select_device', return_value=('cpu', 'int8')), \
             patch.dict(sys.modules, {'faster_whisper': SimpleNamespace(WhisperModel=lambda *args, **kwargs: model)}):
            result = generate_subtitles(str(self.media), ModelSize.TINY, extra_display_time=0.0)
        self.assertEqual(len(result), 2)

    def test_cli_validation_precedes_transcription(self):
        """Verify an existing output file stops the CLI before transcription starts."""
        output = Path(self.temp.name) / 'sub.srt'
        output.write_text('keep me')
        with patch('src.core.transcribe_generator.generate_subtitles') as generate, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(['transcribe', str(self.media), '-o', str(output)]), 1)
            generate.assert_not_called()
        self.assertEqual(output.read_text(), 'keep me')

    def test_cli_check_json_and_exit_status(self):
        """Verify JSON diagnostics and a failing exit status when required CUDA is unavailable."""
        with patch('src.core.device.check_device', return_value=DeviceReport()), contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(['check-device', '--json', '--require-cuda']), 1)
            self.assertIn('"cuda_available": false', output.getvalue())


if __name__ == '__main__':
    unittest.main()
