# TRSCRIBE

Turn audio and video into subtitles from your terminal. Use the interactive menu to choose your settings, or run a single command when you already know what you need.

TRSCRIBE uses Whisper through faster-whisper and supports **SRT**, **VTT**, and **ASS** subtitle files. It runs on CPU or a compatible NVIDIA CUDA GPU.

## Install and start

You need **Python 3.10 or newer**. Run these commands from the project folder.

### First-time setup

On Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
transcribe
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e .
transcribe
```

If you already use this project's `venv312` environment:

```bash
source venv312/bin/activate
transcribe
```

Keep the environment activated when using `transcribe`. Once installed, the command works from any folder.

The first transcription with a model downloads its weights, so it needs an internet connection and may take longer. Downloaded models are cached for later runs.

## Use the interactive menu

Start with no arguments:

```bash
transcribe
```

The start page shows CPU/CUDA availability, the selected model, and the active preset. The initial settings are model **small**, preset **default**, and automatic device selection.

| Option | What it does |
| --- | --- |
| **1. Run** | Ask for the media file, subtitle format, output location, language, and whether to normalize audio; then generate subtitles. |
| **2. Change model-size** | List the available models and let you choose one by name. |
| **3. Change parameters preset** | Select a built-in or saved custom preset. |
| **4. View preset details** | Show a preset's settings and explain each parameter without changing your selection. |
| **5. Create custom preset** | Customize the active preset, save it under a new name, and select it. |
| **q. Quit** | Exit the tool. |

For your first run:

1. Enter `1` to choose **Run**.
2. Enter the path to your audio or video file, such as `/home/user/videos/meeting.mp4`.
3. Press **Enter** to accept the displayed defaults, or enter your own choices.
4. Wait for transcription to finish. The tool prints the saved subtitle location and returns to the menu.

Model and preset selections stay active for the current session. **Ctrl-C** exits the tool.

During a run, stage messages show normalization (when enabled), model loading, and
audio decoding. Transcription displays a progress bar with percentage, processed
audio seconds, and elapsed time in both interactive and command-line mode. Progress
updates as speech segments finish, so it can pause or jump during silence. The bar
tracks transcription, not model downloads or subtitle export. When output is
redirected, the tool writes plain status messages instead of an animated bar.

## Where are subtitles saved?

By default, subtitles are saved **beside the input file**, using its name and the chosen extension:

```text
Input:   /home/user/videos/meeting.mp4
Output:  /home/user/videos/meeting.srt
```

Choose a different location in the interactive prompts or with `-o`:

```bash
transcribe meeting.mp4 -o subtitles/meeting.srt
```

The destination folder must already exist. Interactive mode asks before replacing an existing file; command-line mode requires `--overwrite`. The tool refuses to replace your input media with subtitle output.

## Use a single command

Generate SRT subtitles with automatic language and device selection:

```bash
transcribe "path/to/meeting.mp4"
```

Put paths containing spaces inside quotes.

### Choose a subtitle format

- **SRT**: a simple subtitle format for many video players.
- **VTT**: subtitles commonly used for web video.
- **ASS**: subtitles with a default text style.

```bash
transcribe meeting.mp4 --format vtt
transcribe meeting.mp4 --format ass -o subtitles/meeting.ass
```

Use `--format` to choose the format; changing the output filename alone does not select it.

### Choose a model

```bash
transcribe meeting.mp4 --model medium
```

Available model names: `tiny`, `base`, `small`, `medium`, `medium.en`, `large`, `large-v3`, and `turbo`.

The default is `small`. Model size affects processing time and memory requirements. If you run out of memory, try a smaller model. `medium.en` is intended for English audio.

### Set the language or translate into English

Language detection is automatic unless you provide a code, such as `en`, `ja`, or `es`:

```bash
transcribe meeting.mp4 --language en
transcribe interview.mp4 --language ja --translate
```

`--translate` produces English subtitles from the spoken language. It does not choose an arbitrary target language.

### Normalize audio before transcription

```bash
transcribe meeting.mp4 --normalize
```

Normalization prepares a temporary mono, 16 kHz WAV file before transcription. It also applies duration correction when audio timing metadata is available and accounts for the audio's start offset. Duration correction relies on accurate source metadata.

This converts the audio format; it does not normalize volume. Your original media stays unchanged, and temporary audio is removed afterward. Normalization is **off by default** and requires both **ffmpeg** and **ffprobe** on your PATH. Normal transcription does not require those executables.

You can explicitly disable it with `--no-normalize`.

### Other useful options

| Option | Purpose |
| --- | --- |
| `--vad-filter` | Filter non-speech audio before transcription. |
| `--extra-display-time 0.5` | Keep each subtitle visible for up to 0.5 extra seconds, stopping at the next cue. |
| `--beam-size 10` | Set the number of decoding candidates considered. |
| `--overwrite` | Allow replacement of an existing subtitle file. |
| `--device-index 0` | Choose a CUDA GPU by its visible device index. |
| `--compute-type int8` | Override automatic compute precision; unsupported choices are rejected. |

For all command-line options:

```bash
transcribe --help
transcribe transcribe --help
```

## Use your NVIDIA GPU

Check whether CUDA is ready:

```bash
transcribe check-device
```

The report shows visible CUDA devices, supported compute types, and detected problems. It does not download a model. A successful check does not guarantee that every model will fit in GPU memory.

On Linux, install the optional CUDA runtime libraries into your activated environment from the project folder:

```bash
python -m pip install -e ".[cuda]"
```

You also need a working NVIDIA driver. The Linux launcher automatically finds the pip-installed CUDA libraries. The optional package above only installs runtime libraries on Linux; Windows CUDA setups need the corresponding libraries configured separately.

Choose a device explicitly:

```bash
transcribe meeting.mp4 --device cuda
transcribe meeting.mp4 --device cpu
```

With the default `--device auto`, the tool selects CUDA when its compatibility check passes and otherwise uses CPU. Explicit `--device cuda` reports an error if CUDA is unavailable.

For scripts, `transcribe check-device --json --require-cuda` prints JSON and returns a nonzero exit status when CUDA is unavailable.

## Built-in and custom presets

Presets are selected through the **interactive menu**. The two built-in presets are:

| Preset | Key settings |
| --- | --- |
| **default** | Beam size `10`, no-speech threshold `0.35`. |
| **sensitive** | Beam size `5`, no-speech threshold `0.2`. |

Both disable previous-text conditioning and VAD, use temperature `0.0`, and enable word timestamps. Use **View preset details** to explore the full settings. The `sensitive` name comes from the project's original preset; results depend on the recording.

To create your own:

1. Select a preset to use as your starting point.
2. Choose **5. Create custom preset**.
3. Enter a unique name, such as `interviews` or `lecture_en`.
4. Enter new settings, or press **Enter** to keep each existing value. Enter `-` to clear an initial prompt.

Invalid values are prompted again. The completed preset is saved and selected automatically. Names must start with a lowercase letter and contain only lowercase letters, digits, underscores, or hyphens, up to 40 characters. Existing preset names cannot be replaced through this menu.

Custom presets are saved in `~/.config/trscribe/presets.json`, or `$XDG_CONFIG_HOME/trscribe/presets.json` when configured. They are available on later launches, although each new session starts with the default selection.

**Single-command transcription uses its own defaults and does not automatically apply your interactive preset.**

## Troubleshooting

| Problem | What to try |
| --- | --- |
| `transcribe: command not found` | Activate the environment where you installed the project. If needed, run `python -m pip install -e .` from the project folder. |
| CUDA is unavailable | Run `transcribe check-device` for details and `nvidia-smi` to check driver/GPU visibility. You can use `--device cpu` meanwhile. |
| GPU memory error | Select a smaller model or run on CPU. |
| Normalization cannot find FFmpeg | Install FFmpeg with ffprobe and make both commands available on PATH, or omit `--normalize`. |
| Output already exists | Choose another filename or explicitly allow replacement. |
| Output directory does not exist | Create the folder before starting the run. |
| Custom presets cannot load | Check the reported JSON file for invalid settings. The tool preserves the file instead of overwriting it. |

## Development

See [CONTRIBUTORS.md](CONTRIBUTORS.md) for developer setup, architecture, testing, and contribution guidance.

Run the test suite from the project folder:

```bash
python -m unittest discover -s tests -v
```

Tests do not download models or require a GPU. Audio conversion tests use FFmpeg when installed. The original module entry point, `python -m src`, also remains available.
