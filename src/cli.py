"""Command-line interface for transcribe and python -m src."""

import argparse
from pathlib import Path
import sys


def entrypoint():
    """Prepare CUDA libraries and run the installed command, accepting a media path directly."""
    from src.core.environment import setup

    setup()
    arguments = sys.argv[1:]
    if not arguments:
        from src.interactive import interactive_main

        return interactive_main()
    elif arguments[0] not in {"transcribe", "check-device", "-h", "--help"}:
        arguments = ["transcribe", *arguments]
    return main(arguments)


def parser():
    """Build the command-line parser for transcription and device diagnostics."""
    from src.model.options import ModelSize

    root = argparse.ArgumentParser(description="Transcribe audio/video to SRT, VTT, or ASS using faster-whisper.")
    commands = root.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check-device", help="Check CUDA compatibility without downloading a model")
    check.add_argument("--device-index", type=int, default=0)
    check.add_argument("--json", action="store_true", help="Print a machine-readable report")
    check.add_argument("--require-cuda", action="store_true", help="Exit with status 1 if CUDA is unavailable")

    # command to transcribe audio/video to subtitles
    transcribe = commands.add_parser("transcribe", help="Generate subtitles (first use downloads model weights)")
    transcribe.add_argument("input", type=Path)
    transcribe.add_argument("-o", "--output", type=Path)
    transcribe.add_argument("-f", "--format", choices=["srt", "vtt", "ass"], default="srt")

    # model options
    transcribe.add_argument(
        "--model",
        default=ModelSize.SMALL.value,
        choices=[model.value for model in ModelSize],
    )

    transcribe.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    transcribe.add_argument("--device-index", type=int, default=0)
    transcribe.add_argument("--compute-type", default="auto")
    transcribe.add_argument("--language", help="Whisper language code; default: automatic detection")
    transcribe.add_argument("--beam-size", type=int, default=5)
    transcribe.add_argument("--extra-display-time", type=float, default=0.0)
    transcribe.add_argument("--vad-filter", action="store_true")
    transcribe.add_argument("--translate", action="store_true", help="Translate speech into English")
    transcribe.add_argument("--overwrite", action="store_true")
    transcribe.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Normalize audio to mono 16 kHz WAV before transcription (requires FFmpeg)",
    )
    return root


def main(argv=None):
    """Run the selected CLI command and return its process exit status."""
    args = parser().parse_args(argv)
    try:
        from src.core.device import check_device

        if args.command == "check-device":
            report = check_device(args.device_index)
            if args.json:
                print(report.model_dump_json(indent=2))
            else:
                print(f'CUDA preflight: {"passed" if report.cuda_available else "unavailable"}')
                print(f"Visible CUDA devices: {report.device_count}")
                print(f'CUDA compute types: {", ".join(report.compute_types) or "none"}')
                print(f'CPU compute types: {", ".join(report.cpu_compute_types) or "none"}')
                for issue in report.issues:
                    print(f"- {issue}")
                print(report.note)
            return 1 if args.require_cuda and not report.cuda_available else 0

        from src.core.languages import Lang
        from src.model.options import DeviceType, ExportRequest, ModelSize, TranscriptionRequest, TranscribeOptions

        request = TranscriptionRequest(
            file_input=args.input,
            model_size=ModelSize(args.model),
            device_type=DeviceType(args.device),
            device_index=args.device_index,
            normalize=args.normalize,
            compute_type=args.compute_type,
            lang=Lang(args.language) if args.language else None,
            extra_display_time=args.extra_display_time,
            options=TranscribeOptions(
                beam_size=args.beam_size,
                vad_filter=args.vad_filter,
                task="translate" if args.translate else "transcribe",
            ),
        )
        output = ExportRequest(
            path=args.output or args.input.with_suffix("." + args.format), format=args.format, overwrite=args.overwrite
        )

        return execute_transcription(request, output)

    except (
        ValueError,
        TypeError,
        OSError,
        RuntimeError,
        ImportError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130


def execute_transcription(request, output):
    """Generate and export validated requests shared by CLI and interactive mode."""
    from src.model.options import TranscriptionRequest, ExportRequest
    from src.core.file_manager import FileManager, FileType
    from src.core.transcribe_generator import generate_subtitles

    request = TranscriptionRequest.model_validate(request)
    output = ExportRequest.model_validate(output)
    if output.path.resolve() == request.file_input.resolve():
        raise ValueError("Output must not replace the input media")
    if output.path.exists() and not output.overwrite:
        raise FileExistsError(f"Output exists: {output.path}; use --overwrite to replace it")

    segments = generate_subtitles(
        str(request.file_input),
        request.model_size,
        request.device_type,
        request.lang,
        request.extra_display_time,
        compute_type=request.compute_type,
        device_index=request.device_index,
        normalize=request.normalize,
        **request.options.model_dump(),
    )
    FileManager.export_subtitles(segments, FileType(output.format), output.path, overwrite=output.overwrite)
    print(f"Saved {len(segments)} subtitles to {output.path}")
    return 0
