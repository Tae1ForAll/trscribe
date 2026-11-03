"""Interactive session tests without downloads, external tools, or real inference."""
import ast
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.cli import entrypoint
from src.core.device import DeviceReport
from src.interactive import interactive_main, run_interactive, SessionSettings, show_status
from src.model.options import ModelSize
from src.model.presets import ParameterPreset, preset_options


class InteractiveTests(unittest.TestCase):
    def test_bare_command_opens_interactive_mode(self):
        """Route the bare command to the interactive menu after CUDA setup."""
        with patch('src.core.environment.setup') as setup, \
             patch('src.cli.sys.argv', ['transcribe']), \
             patch('src.interactive.interactive_main', return_value=0) as interactive:
            self.assertEqual(entrypoint(), 0)
            setup.assert_called_once_with()
            interactive.assert_called_once_with()

    def test_presets_match_manual_script_without_executing_it(self):
        """Compare preset values with test_only.py without importing its transcription run."""
        path = Path(__file__).resolve().parents[1] / 'test_only.py'
        tree = ast.parse(path.read_text())
        expected = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                name = node.targets[0].id if isinstance(node.targets[0], ast.Name) else ''
                if name in {'default_params', 'sensitive_params'}:
                    expected[name] = {item.arg: ast.literal_eval(item.value) for item in node.value.keywords}
        for preset in ParameterPreset:
            actual = preset_options(preset).model_dump()
            for name, value in expected[preset.value + '_params'].items():
                self.assertEqual(actual[name], value)
        first = preset_options(ParameterPreset.DEFAULT)
        first.beam_size = 1
        self.assertEqual(preset_options(ParameterPreset.DEFAULT).beam_size, 10)

    def test_changes_persist_and_invalid_choices_recover(self):
        """Retain successful model and preset changes while recovering from invalid input."""
        inputs = ['wrong', '2', 'invalid-model', '2', 'tiny', '3', 'sensitive', '1', 'q']
        report = DeviceReport(cpu_compute_types=['int8'])
        with patch('src.interactive.check_device', return_value=report), \
             patch('builtins.input', side_effect=inputs), \
             patch('src.interactive.run_interactive') as run, \
             contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(interactive_main(), 0)
            settings = run.call_args.args[0]
            self.assertEqual(settings.model_size, ModelSize.TINY)
            self.assertEqual(settings.preset, ParameterPreset.SENSITIVE)
            self.assertIn('TRSCRIBE', output.getvalue())
            self.assertIn('Error:', output.getvalue())
            self.assertIn('Model: tiny', output.getvalue())

    def test_status_reports_gpu_and_cpu(self):
        """Show runtime device availability and the active preset values."""
        for available in (False, True):
            report = DeviceReport(cuda_available=available, device_count=int(available),
                                  compute_types=['float16'] if available else [], cpu_compute_types=['int8'])
            with patch('src.interactive.check_device', return_value=report), contextlib.redirect_stdout(io.StringIO()) as output:
                show_status(SessionSettings())
                text = output.getvalue()
                self.assertIn('CPU: available', text)
                self.assertIn('CUDA GPU: ' + ('available' if available else 'unavailable'), text)
                self.assertIn('Parameter preset: default', text)
                self.assertNotIn('beam_size:', text)
                self.assertNotIn('no_speech_threshold:', text)

    def test_run_uses_validated_settings_and_exports_subtitles(self):
        """Run the shared export workflow with selected presets and normalization enabled."""
        with tempfile.TemporaryDirectory() as directory:
            media = Path(directory) / 'recording.wav'
            media.write_bytes(b'input')
            with patch('builtins.input', side_effect=[str(media), 'vtt', '', 'en', 'y']), \
                 patch('src.core.transcribe_generator.generate_subtitles', return_value=[]) as generate, \
                 contextlib.redirect_stdout(io.StringIO()):
                run_interactive(SessionSettings(model_size=ModelSize.BASE, preset=ParameterPreset.SENSITIVE))
                self.assertEqual(generate.call_args.args[1], ModelSize.BASE)
                self.assertTrue(generate.call_args.kwargs['normalize'])
                self.assertEqual(generate.call_args.kwargs['no_speech_threshold'], 0.2)
                self.assertTrue(generate.call_args.kwargs['word_timestamps'])
                self.assertTrue(media.with_suffix('.vtt').read_text().startswith('WEBVTT'))

    def test_existing_output_requires_yes(self):
        """Keep existing subtitles when the user declines replacement."""
        with tempfile.TemporaryDirectory() as directory:
            media = Path(directory) / 'input.wav'
            media.write_bytes(b'input')
            output = media.with_suffix('.srt')
            output.write_text('keep')
            with patch('builtins.input', side_effect=[str(media), '', '', '', '', 'n']), \
                 patch('src.cli.execute_transcription') as execute, \
                 contextlib.redirect_stdout(io.StringIO()):
                run_interactive(SessionSettings())
                execute.assert_not_called()
                self.assertEqual(output.read_text(), 'keep')

    def test_run_failure_returns_to_menu(self):
        """Recover from a transcription failure without discarding session settings."""
        with patch('src.interactive.check_device', return_value=DeviceReport()), \
             patch('builtins.input', side_effect=['1', 'q']), \
             patch('src.interactive.run_interactive', side_effect=RuntimeError('inference failed')), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(interactive_main(), 0)
            self.assertIn('inference failed', output.getvalue())

    def test_eof_and_interrupt_exit_cleanly(self):
        """Exit without a traceback on closed input or Ctrl-C."""
        for error, expected in [(EOFError(), 0), (KeyboardInterrupt(), 130)]:
            with patch('src.interactive.check_device', return_value=DeviceReport()), \
                 patch('builtins.input', side_effect=error), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(interactive_main(), expected)


if __name__ == '__main__':
    unittest.main()
