"""UTF-8 subtitle serialization."""
from datetime import timedelta
from enum import Enum
from pathlib import Path
import html
import srt
from src.model.options import ExportRequest, SubtitleSegment


class FileType(str, Enum):
    SRT = 'srt'
    VTT = 'vtt'
    ASS = 'ass'


def timestamp(seconds, precision=1000, separator='.'):
    """Format seconds as a subtitle timestamp with millisecond or centisecond precision."""
    ticks = round(seconds * precision)
    whole, fraction = divmod(ticks, precision)
    hours, remainder = divmod(whole, 3600)
    minutes, seconds = divmod(remainder, 60)
    digits = 3 if precision == 1000 else 2
    return f'{hours:02}:{minutes:02}:{seconds:02}{separator}{fraction:0{digits}}'


def render(segments, format):
    """Serialize validated subtitle segments as SRT, WebVTT, or ASS text."""
    if format == 'srt':
        return srt.compose([srt.Subtitle(i, timedelta(seconds=s.start), timedelta(seconds=s.end), s.text.strip())
                            for i, s in enumerate(segments, 1)])
    if format == 'vtt':
        cues = []
        for s in segments:
            text = html.escape(' '.join(s.text.split()), quote=False)
            cues.append(f'{timestamp(s.start)} --> {timestamp(s.end)}\n{text}\n')
        return 'WEBVTT\n\n' + '\n'.join(cues)
    header = '''[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,0,2,40,40,40,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
'''
    lines = []
    for s in segments:
        # Neutralize override tags and backslash commands in transcribed text.
        text = s.text.strip().replace('\\', '＼').replace('{', '｛').replace('}', '｝')
        text = text.replace('\r\n', '\n').replace('\r', '\n').replace('\n', r'\N')
        lines.append(f'Dialogue: 0,{timestamp(s.start, 100)},{timestamp(s.end, 100)},Default,,0,0,0,,{text}\n')
    return header + ''.join(lines)


class FileManager:
    @staticmethod
    def export_subtitles(segments, format: FileType, path=None, *, overwrite=False):
        """Validate and write subtitle segments, requiring opt-in to overwrite an existing file."""
        if path is None:
            raise ValueError('Please enter an output path')
        request = ExportRequest(path=Path(path), format=format.value if isinstance(format, FileType) else format,
                                overwrite=overwrite)
        validated = [SubtitleSegment.model_validate(s, from_attributes=True) for s in segments]
        data = render(validated, request.format)
        with request.path.open('w' if request.overwrite else 'x', encoding='utf-8', newline='\n') as stream:
            stream.write(data)
        return request.path

    @staticmethod
    def import_subtitles(file_path):
        """Read an SRT file and validate cue timings before returning parsed subtitles."""
        from src.model.options import TranscriptionRequest
        request = TranscriptionRequest(file_input=Path(file_path))
        with request.file_input.open(encoding='utf-8-sig') as stream:
            subtitles = list(srt.parse(stream.read()))
        for subtitle in subtitles:
            SubtitleSegment(start=subtitle.start.total_seconds(), end=subtitle.end.total_seconds(), text=subtitle.content)
        return subtitles

    import_subtites = import_subtitles  # Compatibility with the original misspelling.

    @staticmethod
    def write_srt(segments, output_path='output.srt', *, overwrite=False):
        """Export subtitle segments to a UTF-8 SRT file."""
        return FileManager.export_subtitles(segments, FileType.SRT, output_path, overwrite=overwrite)

    @staticmethod
    def write_vtt(segments, output_path='output.vtt', *, overwrite=False):
        """Export subtitle segments to a UTF-8 WebVTT file."""
        return FileManager.export_subtitles(segments, FileType.VTT, output_path, overwrite=overwrite)

    @staticmethod
    def write_ass(segments, output_path='output.ass', *, overwrite=False):
        """Export subtitle segments to a UTF-8 ASS file with a default style."""
        return FileManager.export_subtitles(segments, FileType.ASS, output_path, overwrite=overwrite)
