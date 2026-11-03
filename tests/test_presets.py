"""Custom preset validation, persistence, and interactive selection tests."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pydantic import ValidationError
from src.core.preset_store import CustomPresets, load_presets, save_presets
from src.interactive import SessionSettings, create_preset, view_preset, change_model, change_preset, run_interactive
from src.model.options import TranscribeOptions, ModelSize


class PresetTests(unittest.TestCase):
    def setUp(self):
        """Keep preset writes inside a temporary config directory for each test."""
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'presets.json'
        self.path_patch = patch('src.core.preset_store.preset_path', return_value=self.path)
        self.path_patch.start()
        self.addCleanup(self.path_patch.stop)

    def test_create_validates_retries_saves_and_selects(self):
        """Retry invalid numeric input and persist the newly selected custom options."""
        answers = ['my-preset', '0', '7', '', 'yes', 'nan', '0.25', '', '', 'translate', 'Technical terms']
        with patch('builtins.input', side_effect=answers), contextlib.redirect_stdout(io.StringIO()) as output:
            settings = create_preset(SessionSettings())
        self.assertEqual(settings.preset, 'my-preset')
        self.assertEqual(settings.options_for().beam_size, 7)
        self.assertEqual(settings.options_for().task, 'translate')
        self.assertTrue(settings.options_for().vad_filter)
        self.assertIn('Invalid beam_size', output.getvalue())
        restored = SessionSettings(preset='my-preset', custom_presets=load_presets())
        self.assertEqual(restored.options_for(), settings.options_for())

    def test_reserved_and_invalid_names_rejected(self):
        """Protect built-in names and reject malformed or path-like custom names."""
        for name in ['default', 'sensitive', '../bad', '', 'a' * 41, 'White space']:
            with self.subTest(name=name), self.assertRaises(ValidationError):
                CustomPresets(presets={name: TranscribeOptions()})
        self.assertFalse(self.path.exists())

    def test_duplicate_name_keeps_existing_preset(self):
        """Reject creating an existing preset without modifying its file."""
        save_presets({'mine': TranscribeOptions(beam_size=9)})
        original = self.path.read_text()
        with patch('builtins.input', return_value='mine'), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, 'already exists'):
                create_preset(SessionSettings())
        self.assertEqual(self.path.read_text(), original)

    def test_malformed_store_is_not_overwritten(self):
        """Reject corrupt persisted settings while preserving the original file."""
        self.path.write_text('{broken')
        with self.assertRaises(ValidationError):
            save_presets({'mine': TranscribeOptions()})
        self.assertEqual(self.path.read_text(), '{broken')

    def test_detail_view_describes_custom_settings_without_changing_selection(self):
        """Inspect any preset without switching the active preset."""
        settings = SessionSettings(custom_presets={'mine': TranscribeOptions(beam_size=12)})
        with patch('builtins.input', side_effect=['mine', '']), contextlib.redirect_stdout(io.StringIO()) as output:
            view_preset(settings)
        self.assertIn('beam_size: 12', output.getvalue())
        self.assertIn('Number of decoding candidates', output.getvalue())
        self.assertEqual(settings.preset, 'default')

    def test_switching_preserves_custom_definitions(self):
        """Keep saved custom options when changing the active model or preset."""
        settings = SessionSettings(custom_presets={'mine': TranscribeOptions(beam_size=12)})
        with patch('builtins.input', side_effect=['mine', 'tiny']), contextlib.redirect_stdout(io.StringIO()):
            settings = change_preset(settings)
            settings = change_model(settings)
        self.assertEqual(settings.model_size, ModelSize.TINY)
        self.assertEqual(settings.preset, 'mine')
        self.assertEqual(settings.options_for().beam_size, 12)

    def test_custom_settings_reach_transcription(self):
        """Forward every custom option through the interactive run workflow."""
        media = Path(self.temp.name) / 'input.wav'
        media.write_bytes(b'input')
        options = TranscribeOptions(beam_size=13, temperature=0.3, initial_prompt='Names')
        settings = SessionSettings(preset='mine', custom_presets={'mine': options})
        with patch('builtins.input', side_effect=[str(media), '', '', '', '']), \
             patch('src.core.transcribe_generator.generate_subtitles', return_value=[]) as generate, \
             contextlib.redirect_stdout(io.StringIO()):
            run_interactive(settings)
        for name, value in options.model_dump().items():
            self.assertEqual(generate.call_args.kwargs[name], value)

    def test_incomplete_creation_does_not_save(self):
        """Leave storage untouched if input closes before creation is complete."""
        with patch('builtins.input', side_effect=['mine', EOFError()]), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(EOFError):
                create_preset(SessionSettings())
        self.assertFalse(self.path.exists())


if __name__ == '__main__':
    unittest.main()
