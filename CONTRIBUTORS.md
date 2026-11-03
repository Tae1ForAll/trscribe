# Contributing to TRSCRIBE

This guide is for developers working on the application. For installation and everyday usage, see [README.md](README.md). Read [AGENTS.md](AGENTS.md) before making changes.

## Development setup

Use Python 3.10 or newer. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

On Windows, create the environment with `python -m venv .venv` and activate it with `.venv\Scripts\Activate.ps1` in PowerShell. If you already use `venv312`, activate that environment instead.

Editable installation makes source changes available to the `transcribe` command immediately. Reinstall after changing package dependencies or command entry points in `pyproject.toml`.

For Linux CUDA development, install the existing optional dependencies:

```bash
python -m pip install -e ".[cuda]"
transcribe check-device
```

FFmpeg and ffprobe must be on PATH to exercise audio normalization. They are not needed for the standard transcription path. A GPU and downloaded model weights are not required for the automated test suite.

Use **pip** for package management. As required by `AGENTS.md`, ask for confirmation before adding a new dependency. Runtime dependencies and optional extras belong in `pyproject.toml`; `requirements.txt` retains the older environment's broader dependency list.

## Project layout

The Python package is currently named `src`; this is also the import path. Setuptools includes `src` and its subpackages explicitly.

| Path | Responsibility |
| --- | --- |
| `pyproject.toml` | Package metadata, dependencies, and the `transcribe` entry point. |
| `src/__main__.py` | Support for `python -m src`. |
| `src/cli.py` | Startup, argument parsing, and the shared transcription/export workflow. |
| `src/interactive.py` | Terminal panels, prompts, session settings, and custom preset creation. |
| `src/model/options.py` | Pydantic request models, subtitle cues, and model/device enums. |
| `src/model/presets.py` | Built-in `default` and `sensitive` preset definitions. |
| `src/core/environment.py` | Linux CUDA library search-path setup before backend use. |
| `src/core/device.py` | CTranslate2 device diagnostics and compute-type selection. |
| `src/core/transcribe_generator.py` | Whisper inference, cue validation, timing alignment, and display extension. |
| `src/core/progress.py` | Terminal progress based on media duration and segment timestamps, with plain status output when redirected. |
| `src/core/audio.py` | Optional FFmpeg normalization and temporary audio lifecycle. |
| `src/core/file_manager.py` | SRT, VTT, and ASS serialization plus SRT import. |
| `src/core/preset_store.py` | Validation, loading, and atomic saving of custom presets. |
| `src/core/languages.py` | Supported language-code enum. |
| `tests/` | Automated tests using the standard-library `unittest` framework. |

Root scripts such as `repaire.py`, `fixformat.py`, and `test_only.py` are standalone utilities or manual examples. Do not import them into the application: some perform file operations or transcription at import time. Reusable application code belongs in `src/`.

## How a transcription runs

1. `src.cli.entrypoint()` prepares the Linux CUDA library search path. With no arguments, it opens interactive mode; otherwise it routes to the CLI parser.
2. The CLI or interactive prompts construct validated `TranscriptionRequest` and `ExportRequest` objects.
3. `execute_transcription()` checks output protection before starting expensive work.
4. `generate_subtitles()` selects a device and compute type, optionally prepares normalized audio, loads the Whisper model, and consumes its segment iterator.
5. Cues are validated, aligned to the source timeline when normalization provides an offset, and extended by the requested display time.
6. `FileManager` writes the selected subtitle format. Temporary normalized audio is removed when its context exits.

Both interfaces use `execute_transcription()` so fixes to output protection and export behavior apply consistently.

### Important behavior to preserve

- All input data must pass through Pydantic models before processing. `ValidatedModel` uses strict types, forbids unknown fields, and rejects non-finite numbers. External ffprobe metadata has explicit parsing rules and ignores unrelated metadata sections.
- Existing subtitle files require explicit overwrite permission. Never allow subtitle output to replace the source media.
- Whisper returns lazy segments. Consume them **inside** the `prepared_audio()` context so temporary files remain available throughout inference.
- `--device auto` may fall back to CPU during preflight. Explicit CUDA requests fail when unavailable; inference errors do not trigger an automatic second run on CPU.
- Device preflight does not prove that a model will fit in memory or complete inference.
- Linux library-path setup can restart Python with `os.execve`. Keep it in startup code and preserve command arguments. Mock it in unit tests rather than allowing tests to restart the interpreter.
- Normalization preserves the original file. Duration correction depends on source audio metadata; missing audio duration must not be replaced with an assumed video duration.

## Defaults and presets

Interactive mode starts with model `small`, automatic device selection, and the `default` preset. Its parameter values intentionally differ from the standalone CLI defaults.

The Python `generate_subtitles()` API also has compatibility defaults: CPU when no device is supplied, and one second of extra display time. The CLI and interactive workflows pass their own settings explicitly.

Custom presets are validated `TranscribeOptions` collections stored in `~/.config/trscribe/presets.json`, or under `$XDG_CONFIG_HOME` when set. The store protects built-in names and uses atomic file replacement. A malformed existing file must be reported without overwriting it.

Tests that exercise persistence should patch the storage path into a temporary directory. Never write tests against a developer's real preset file.

## Extending the application

### Add a model

Add the supported identifier to `ModelSize` in `src/model/options.py`. CLI choices and the interactive model list derive from the enum. Check that the installed faster-whisper version accepts the identifier, then update the model list in the user README.

### Add a transcription parameter

1. Define its type, default, and constraints in `TranscribeOptions`.
2. Add CLI arguments if it should be available in single-command mode.
3. Update interactive preset creation to parse its input type, and add its explanation to `PARAMETER_HELP`.
4. Update built-in presets if needed and decide how older saved presets should receive the new default.
5. Verify that the value reaches `model.transcribe()` and that invalid values are rejected.

### Add or adjust a preset

Edit `ParameterPreset` and `preset_options()` for built-in presets. Return fresh validated options so editing a custom preset cannot mutate a built-in definition.

Tests verify the original two presets against explicit expected values, independently
of manual scripts. If you intentionally change those definitions, update the relevant
test expectations and user documentation together. Newly added built-ins need their own coverage.

### Add a subtitle format

Update `FileType`, the allowed export formats, serialization, CLI choices, and interactive format prompts. Verify Unicode text, timestamps, format-specific escaping, and overwrite protection.

### Change terminal screens

Use `print_frame()` for device, configuration, and detail panels. Keep the main menu as a plain list. The start page shows the selected model and preset name; full parameter values belong in the preset-details view.

Keep prompts usable without color support and make long content wrap within the terminal width. Preserve blank-input defaults, error recovery, and clean EOF/Ctrl-C exits.

## Testing and validation

Run all tests:

```bash
python -m unittest discover -s tests -v
```

Run a focused suite:

```bash
python -m unittest discover -s tests -p 'test_presets.py' -v
```

| Test file | Coverage |
| --- | --- |
| `test_transcriber.py` | Request validation, device selection, subtitle export, startup, and CLI behavior. |
| `test_audio.py` | Normalization, duration correction, offsets, cleanup, and FFmpeg failures. |
| `test_interactive.py` | Menu routing, preset selection, prompts, run failures, and session exits. |
| `test_presets.py` | Custom preset validation, persistence, inspection, and use during transcription. |
| `test_progress.py` | Progress updates, unknown durations, completion, interruption, and redirected output. |

Inference and GPU responses are mocked. Real audio-conversion tests use generated fixtures and run only when FFmpeg tools are available. Automated tests do not establish real speech-recognition quality or GPU inference success.

Useful checks after changing Python code:

```bash
python -m compileall -q src
python -m pip check
git diff --check
```

For a manual interface check, launch `transcribe`, change a model or preset, inspect details, and quit. For a real inference check, use a short media file you are allowed to process and a distinct output path:

```bash
transcribe sample.wav --model tiny --device cpu -o sample-check.srt
```

That command may download model weights. Validate CUDA separately on a machine where the GPU is accessible. A restricted execution environment can hide a working host GPU, so compare its results with diagnostics from a normal terminal before concluding the driver is broken.

## Preparing a contribution

- Keep application code in `src/` and tests in `tests/`.
- Add a brief explanatory docstring to every new function.
- Preserve existing local changes and keep edits focused on the requested behavior.
- Add meaningful tests for new behavior and failure paths; documentation-only edits do not need new tests.
- Update [README.md](README.md) when user commands or features change, and this guide when architecture or development workflow changes.
- Describe the problem, the resulting behavior, and the checks performed in your change summary. Mention any hardware or inference checks that remain unverified.

Do not commit virtual environments, downloaded models, private media, generated subtitles, or personal preset files.
