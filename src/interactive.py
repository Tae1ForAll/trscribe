"""Interactive terminal menu for configuring and running subtitle generation."""

from pathlib import Path
import shutil
import textwrap
from typing import Literal

from pydantic import Field, model_validator
from src.core.device import check_device
from src.core.preset_store import CustomPresets, load_presets, save_presets, preset_path
from src.core.languages import Lang
from src.model.options import (
    DeviceType,
    ExportRequest,
    ModelSize,
    TranscriptionRequest,
    TranscribeOptions,
    ValidatedModel,
)
from src.model.presets import ParameterPreset, preset_options

BANNER = r"""
 TTTTT  RRRR    SSSS   CCCC  RRRR   IIIII  BBBB   EEEEE
   T    R   R  S      C      R   R    I    B   B  E
   T    RRRR    SSS   C      RRRR     I    BBBB   EEEE
   T    R  R       S  C      R  R     I    B   B  E
   T    R   R  SSSS    CCCC  R   R  IIIII  BBBB   EEEEE
"""


class SessionSettings(ValidatedModel):
    model_size: ModelSize = ModelSize.SMALL
    preset: str = ParameterPreset.DEFAULT.value
    custom_presets: dict[str, TranscribeOptions] = Field(default_factory=dict)

    @model_validator(mode="after")
    def known_preset(self):
        """Validate custom definitions and require the selected preset to exist."""
        CustomPresets(presets=self.custom_presets)
        if self.preset not in [preset.value for preset in ParameterPreset] and self.preset not in self.custom_presets:
            raise ValueError(f"Unknown preset: {self.preset}")
        return self

    def options_for(self, name: str | None = None) -> TranscribeOptions:
        """Return an independent copy of the requested built-in or custom preset."""
        name = name or self.preset
        if name in self.custom_presets:
            return self.custom_presets[name].model_copy(deep=True)
        return preset_options(ParameterPreset(name))


class PromptText(ValidatedModel):
    text: str = Field(max_length=4096)


class MenuChoice(ValidatedModel):
    value: Literal["1", "2", "3", "4", "5", "q"]


class YesNo(ValidatedModel):
    value: Literal["y", "yes", "n", "no"]


def prompt(message: str, default: str = "") -> str:
    """Read a bounded text response, using the supplied default for blank input."""
    text = PromptText(text=input(message)).text.strip()
    return text or default


def prompt_path(message: str, default: str = "") -> Path:
    """Read a path, accepting home expansion and paired quotes around spaces."""
    text = prompt(message, default)
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        text = text[1:-1]
    return Path(text).expanduser()


def print_frame(title: str, lines: list[str]):
    """Print a titled star-bordered panel, wrapping content to the terminal width."""
    width = max(12, min(76, shutil.get_terminal_size(fallback=(76, 24)).columns))
    inner = width - 4
    border = "*" * width
    print("\n" + border)
    for line in textwrap.wrap(title, width=inner):
        print(f"* {line.center(inner)} *")
    print(border)
    for line in lines:
        for wrapped in textwrap.wrap(line, width=inner, subsequent_indent="  ") or [""]:
            print(f"* {wrapped.ljust(inner)} *")
    print(border)


def show_status(settings: SessionSettings):
    """Display framed device status, active configuration, and menu actions."""
    report = check_device()
    selected = "CUDA (device 0)" if report.cuda_available else "CPU" if report.cpu_compute_types else "none"
    device_lines = [
        f'CPU: {"available" if report.cpu_compute_types else "unavailable"}',
        f'CUDA GPU: {"available" if report.cuda_available else "unavailable"} ({report.device_count} visible)',
        f"Device selection: auto -> {selected}",
        f'CUDA compute types: {", ".join(report.compute_types) or "none"}',
    ]
    if report.issues:
        device_lines += ["", "Diagnostics:"] + [f"- {issue}" for issue in report.issues]

    print_frame("AVAILABLE DEVICES", device_lines)

    print_frame(
        "CURRENT CONFIGURATION",
        [
            f"Model: {settings.model_size.value}",
            f"Parameter preset: {settings.preset}",
            "",
            "The model loads when Run starts.",
        ],
    )
    print("\nMAIN MENU")
    print("  1. Run")
    print("  2. Change model-size")
    print("  3. Change parameters preset")
    print("  4. View preset details")
    print("  5. Create custom preset")
    print("\n  q. Quit")


def change_model(settings: SessionSettings) -> SessionSettings:
    """Choose a model by its enum value and return validated session settings."""
    print_frame(
        "CHANGE MODEL-SIZE",
        [
            f"Current model: {settings.model_size.value}",
            "",
            "Available: " + ", ".join(model.value for model in ModelSize),
            "",
            "Enter a model name, or press Enter to keep the current model.",
        ],
    )
    selected = ModelSize(prompt(f"Model [{settings.model_size.value}]: ", settings.model_size.value))
    return SessionSettings(model_size=selected, preset=settings.preset, custom_presets=settings.custom_presets)


def change_preset(settings: SessionSettings) -> SessionSettings:
    """Choose a built-in or custom preset while retaining the selected model."""
    descriptions = []
    for name in [*[preset.value for preset in ParameterPreset], *settings.custom_presets]:
        options = settings.options_for(name)
        descriptions.append(
            f"{name:<9} - beam size {options.beam_size}; " f"no-speech threshold {options.no_speech_threshold}"
        )
    print_frame(
        "CHANGE PARAMETERS PRESET",
        [
            f"Current preset: {settings.preset}",
            "",
            *descriptions,
            "",
            "Enter a preset name, or press Enter to keep the current preset.",
        ],
    )
    selected = prompt(f"Preset [{settings.preset}]: ", settings.preset)
    return SessionSettings(model_size=settings.model_size, preset=selected, custom_presets=settings.custom_presets)


PARAMETER_HELP = {
    "beam_size": "Number of decoding candidates (integer >= 1).",
    "condition_on_previous_text": "Use earlier transcription as context.",
    "vad_filter": "Filter non-speech audio before transcription.",
    "no_speech_threshold": "Silence threshold (0-1); lower values match the sensitive preset.",
    "temperature": "Decoding sampling temperature (0-1).",
    "word_timestamps": "Calculate timestamps for individual words.",
    "task": "transcribe preserves the language; translate outputs English.",
    "initial_prompt": "Optional text to guide vocabulary or style.",
}


def view_preset(settings: SessionSettings):
    """Show a selected preset's complete settings and brief parameter explanations."""
    names = [*[preset.value for preset in ParameterPreset], *settings.custom_presets]
    print_frame("VIEW PRESET DETAILS", ["Available: " + ", ".join(names)])
    name = prompt(f"Preset [{settings.preset}]: ", settings.preset)
    options = settings.options_for(name)
    lines = []
    for key, value in options.model_dump().items():
        lines.extend([f"{key}: {value}", PARAMETER_HELP[key], ""])
    print_frame(f"PRESET: {name}", lines)
    prompt("Press Enter to return to the menu: ")


def create_preset(settings: SessionSettings) -> SessionSettings:
    """Collect validated overrides, save a custom preset, and select it for this session."""
    print_frame(
        "CREATE CUSTOM PRESET",
        [
            f"Starting from: {settings.preset}",
            "Name: lowercase letters, digits, underscores or hyphens; start with a letter.",
            "Press Enter to keep a value. Enter - to clear the initial prompt.",
        ],
    )
    name = prompt("New preset name: ")
    base = settings.options_for()
    CustomPresets(presets={name: base})
    if name in settings.custom_presets or name in load_presets():
        raise ValueError(f"Preset already exists: {name}. Choose a new name.")
    values = base.model_dump()
    for key, current in list(values.items()):
        while True:
            print(PARAMETER_HELP[key])
            raw = prompt(f"{key} [{current}]: ")
            if not raw:
                break

            try:
                if isinstance(current, bool):
                    if raw.lower() not in {"true", "false", "yes", "no", "y", "n"}:
                        raise ValueError("Enter true/false or yes/no")
                    value = raw.lower() in {"true", "yes", "y"}
                elif key == "beam_size":
                    value = int(raw)
                elif key in {"temperature", "no_speech_threshold"}:
                    value = float(raw)
                else:
                    value = None if key == "initial_prompt" and raw == "-" else raw
                values = TranscribeOptions(**{**values, key: value}).model_dump()
                break
            except ValueError as exc:
                print(f"Invalid {key}: {exc}")

    custom = {**settings.custom_presets, name: TranscribeOptions(**values)}
    result = SessionSettings(model_size=settings.model_size, preset=name, custom_presets=custom)
    save_presets(custom)
    print(f'Saved and selected preset "{name}". File: {preset_path()}')
    return result


def run_interactive(settings: SessionSettings):
    """Collect run inputs, validate them, and generate subtitles with the active preset."""
    from src.cli import execute_transcription

    print_frame(
        "RUN SUBTITLE GENERATION",
        [
            f"Model: {settings.model_size.value}",
            f"Parameter preset: {settings.preset}",
            "",
            "Enter the input file and output preferences below.",
            "Press Enter to accept any value shown in brackets.",
        ],
    )
    file_input = prompt_path("Input audio/video path: ")

    # Reject missing media before asking for the rest of the run settings.
    TranscriptionRequest(file_input=file_input)
    format = prompt("Subtitle format [srt] (srt/vtt/ass): ", "srt")
    default_output = file_input.with_suffix("." + format) if format in {"srt", "vtt", "ass"} else file_input
    output = ExportRequest(path=prompt_path(f"Output path [{default_output}]: ", str(default_output)), format=format)
    language = prompt("Language code [auto]: ", "auto")
    normalize = YesNo(value=prompt("Normalize audio before transcription? [y/N]: ", "n").lower()).value
    request = TranscriptionRequest(
        file_input=file_input,
        model_size=settings.model_size,
        device_type=DeviceType.AUTO,
        lang=None if language == "auto" else Lang(language),
        normalize=normalize in {"y", "yes"},
        options=settings.options_for(),
    )
    if output.path.resolve() == request.file_input.resolve():
        raise ValueError("Output must not replace the input media")
    if output.path.exists():
        answer = YesNo(value=prompt("Output exists. Overwrite? [y/N]: ", "n").lower()).value
        if answer not in {"y", "yes"}:
            print("Run cancelled; existing output kept.")
            return
        output = ExportRequest(path=output.path, format=output.format, overwrite=True)
    print(f"Running {settings.model_size.value} with the {settings.preset} preset...")
    print("Model weights may download on first use.")
    execute_transcription(request, output)


def interactive_main() -> int:
    """Run the interactive session until quit, EOF, or keyboard interruption."""
    if shutil.get_terminal_size(fallback=(76, 24)).columns >= 54:
        print(BANNER)
    print_frame("TRSCRIBE", ["Audio & video subtitles", "Configure. Transcribe. Export."])
    try:
        settings = SessionSettings(custom_presets=load_presets())
    except (ValueError, OSError) as exc:
        print(f"Cannot load custom presets from {preset_path()}: {exc}")
        return 1
    try:
        while True:
            try:
                show_status(settings)
                choice = MenuChoice(value=prompt("\n  Choose an option > ").lower()).value
                if choice == "q":
                    print("Goodbye.")
                    return 0
                if choice == "1":
                    run_interactive(settings)
                elif choice == "2":
                    settings = change_model(settings)
                elif choice == "3":
                    settings = change_preset(settings)
                elif choice == "4":
                    view_preset(settings)
                elif choice == "5":
                    settings = create_preset(settings)
            except (ValueError, TypeError, OSError, RuntimeError, ImportError) as exc:
                print(f"Error: {exc}")
                print("Please try again; your model and preset selection are retained.")

    except EOFError:
        print("\nGoodbye.")
        return 0

    except KeyboardInterrupt:
        print("\nInterrupted. Exiting TRSCRIBE.")
        return 130
