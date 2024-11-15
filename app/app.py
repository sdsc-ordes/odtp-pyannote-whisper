import os
import argparse
from typing import Any, Optional, TextIO, List
from pyannote.audio import Pipeline, Audio
import whisper
from whisper.utils import WriteSRT, WriteVTT
from whisper import Whisper
import torch
from math import ceil, floor
import soundfile as sf
import librosa
import json
from dataclasses import dataclass, asdict
from typing import List
from jsonschema import validate, ValidationError


@dataclass
class Segment:
    start: float
    end: float
    text: str
    speaker: str
    language: str

def generate_segments(transcription_data, speaker, language) -> List[Segment]:
    segments = []
    for item in transcription_data:
        segment = Segment(
            start=item['start'],
            end=item['end'],
            text=item['text'],
            speaker=speaker,
            language=language
        )
        segments.append(segment)
    return segments


schema = {
    "type": "object",
    "properties": {
        "segments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start": {"type": "number"},
                    "end": {"type": "number"},
                    "text": {"type": "string"},
                    "speaker": {"type": "string"},
                    "language": {"type": "string"}
                },
                "required": ["start", "end", "text", "speaker", "language"]
            }
        }
    },
    "required": ["segments"]
}

def validate_json(json_data: str) -> bool:
    try:
        data = json.loads(json_data)
        validate(instance=data, schema=schema)
        return True
    except ValidationError as e:
        print(f"Validation error: {e}")
        return False

def diarize_audio(HF_AUTH_TOKEN, AUDIO_FILE):
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=HF_AUTH_TOKEN)
    # Send pyannote pipeline to GPU (when available)
    device: str = ""
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    pipeline.to(torch.device(device))
    print(f"Diarize audio on {device}")  
    io = Audio(mono='downmix', sample_rate=16000)
    waveform, sample_rate = io(AUDIO_FILE)
    diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate})
    return diarization, waveform, sample_rate

class AppendResultsMixin:
    """Class to return srt or vtt file path and open mode of write or append
    to allow incremental writing.
    """
    first_call: bool = True
    output_path: str = ''
    
    def get_path_and_open_mode(self, *, audio_path: str, dir: str, ext: str) -> tuple[str, str]:
        mode: str
        if self.first_call:
            audio_basename = os.path.basename(audio_path)
            audio_basename = os.path.splitext(audio_basename)[0]
            self.output_path: str = os.path.join(dir, audio_basename + "." + ext)
            self.first_call = False
            mode = 'w' # open for write initially
        else:
            mode = 'a' # open for append after
        return self.output_path, mode

class WriteSRTIncremental(AppendResultsMixin, WriteSRT):
    """Incrementally create an SRT file with multiple calls appending new entries
      to the file.
    """
    srt_index: int = 1  # Index for SRT blocks retained across multiple calls

    def __init__(self, output_dir: Optional[str] = None):
        super().__init__(output_dir=output_dir)
        self.extension = '.srt'
        self.srt_index = 1  # Move to instance variable

    def __call__(
        self,
        result: dict,
        audio_path: str,
        speaker: str,
        start_base: float,
        options: Optional[dict] = None,
        output_path: Optional[str] = None,
        **kwargs,
    ):
        if output_path:
            path = output_path
            mode = 'a' if os.path.exists(path) else 'w'
        else:
            audio_dir = os.path.dirname(audio_path)
            output_dir = self.output_dir if self.output_dir else audio_dir
            audio_basename = os.path.splitext(os.path.basename(audio_path))[0]
            path = os.path.join(output_dir, f"{audio_basename}{self.extension}")
            mode = 'a' if os.path.exists(path) else 'w'

        with open(path, mode, encoding="utf-8") as f:
            self.write_result(result, f, speaker, start_base, options=options, **kwargs)

    def write_result(
        self,
        result: dict,
        file: TextIO,
        speaker: str,
        start_base: float,
        options: Optional[dict] = None,
        **kwargs,
    ):
        for segment in result['segments']:
            start = self.format_timestamp(segment['start'])
            end = self.format_timestamp(segment['end'])
            text = f"[{speaker}]: {segment['text']}"
            print(f"{self.srt_index}\n{start} --> {end}\n{text}\n", file=file, flush=True)
            self.srt_index += 1  

class WriteVTTIncremental(AppendResultsMixin, WriteVTT):
    """Incrementally create a VTT file with multiple calls appending new entries
      to the file.
    """
    def __call__(
        self,
        result: dict,
        audio_path: str,
        speaker: str,
        start: float,
        options: Optional[dict] = None,
        output_path: Optional[str] = None,
        **kwargs,
    ):
        if output_path:
            path = output_path
            mode = 'a' if os.path.exists(path) else 'w'
        else:
            path, mode = self.get_path_and_open_mode(
                audio_path=audio_path,
                dir=".",
                ext="vtt"
            )
        with open(path, mode, encoding="utf-8") as f:
            self.write_result(result, file=f, options=options, **kwargs)

    def write_result(
        self,
        result: dict,
        file: TextIO,
        options: Optional[dict] = None,
        **kwargs,
    ):
        if file.tell() == 0:
            print("WEBVTT\n", file=file)

        for segment in result['segments']:
            start = self.format_timestamp(segment['start'])
            end = self.format_timestamp(segment['end'])
            text = f"[{segment.get('speaker', 'unknown')}]: {segment['text']}"
            print(f"{start} --> {end}\n{text}\n", file=file, flush=True)

class SegmentsJSONWriter(AppendResultsMixin):
    """Incrementally write segments to a JSON file."""
    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = output_dir  # Now optional
        self.first_call = True
        self.output_path = ''  # Will store the output file path

    def __call__(
        self,
        segments: List[Segment],
        audio_path: str,
        output_path: Optional[str] = None,
    ):
        if output_path:
            # Use the provided output path directly
            path = output_path
            mode = 'a' if os.path.exists(path) else 'w'
            self.output_path = path
        else:
            if not self.output_path:
                audio_basename = os.path.splitext(os.path.basename(audio_path))[0]
                # Use output_dir if provided, else use the directory of audio_path
                dir = self.output_dir if self.output_dir else os.path.dirname(audio_path)
                self.output_path = os.path.join(dir, audio_basename + ".json")
            path = self.output_path
            mode = 'a' if not self.first_call else 'w'

        with open(path, mode, encoding='utf-8') as f:
            if self.first_call:
                # Start the JSON structure
                f.write('{"segments": [\n')
            else:
                f.write(',\n')
            for idx, segment in enumerate(segments):
                if idx > 0:
                    f.write(',\n')
                json.dump(asdict(segment), f, ensure_ascii=False, indent=2)
            self.first_call = False

    def finalize(self):
        """Call this method after all segments have been written to close the JSON array."""
        if self.output_path:
            with open(self.output_path, 'a', encoding='utf-8') as f:
                f.write('\n  ]\n}\n')

    def close(self):
        with open(self.output_path, 'a', encoding='utf-8') as f:
            f.write('\n]}\n')

class WhisperFacade:
    wmodel: Whisper

    def __init__(self, model:str, *, quantize=False) -> None:
        """Load the Whisper model and optionally quantize."""
        print("Initialize whisper")
        whisper_model = whisper.load_model(model)
        if quantize:
            print("Quantize")
            DTYPE = torch.qint8
            qmodel: Whisper = torch.quantization.quantize_dynamic(
                        whisper_model, {torch.nn.Linear}, dtype=DTYPE)
            del whisper_model
            self.wmodel = qmodel
        else:
            self.wmodel = whisper_model

    def _set_timing_for(self, segment: dict[str, float],  # simplified typing
                        offset: float) -> None:
        """For speech fragments in different parts of an audio file, patch the 
        whisper segment and word timing using the offset (typically the diarization offset)
        in seconds. This makes the timing accurate for subtitles when multiple 
        calls to whisper are used for various parts of the audio.
        """
        s = segment
        s['start'] += offset 
        s['end']   += offset
        # Update word start/stop times, if present
        if 'words' in s:
            w: dict[str, float] # simplified typing
            for w in s['words']: # type: ignore
                w['start'] += offset
                w['end'] += offset

    def load_audio(self, file_path: str):
        self.audio = whisper.load_audio(file_path)
    
    def transcribe(self, *, start: float, end: float, options: dict[str, Any] ) -> dict[str, Any]:
        """Transcribe from start time to end time (both in seconds)."""
        SAMPLE_RATE = 16_000 # 16kHz audio
        start_index = floor(start * SAMPLE_RATE)
        end_index = ceil(end * SAMPLE_RATE)
        audio_segment = self.audio[start_index:end_index]
        result = whisper.transcribe(self.wmodel, audio_segment, **options)
        #
        segments = result['segments']
        s: dict[str, float] # simplified typing
        for s in segments: # type: ignore
            self._set_timing_for(segment=s, offset=start)
        return result

def clip_audio(audio_file_path, sample_rate, start, end, output_path):
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load the audio file
    waveform, sr = librosa.load(audio_file_path, sr=sample_rate, mono=True)
    
    # Calculate start and end samples
    start_sample = int(start * sr)
    end_sample = int(end * sr)
    
    # Write the audio segment to the output path
    sf.write(output_path, waveform[start_sample:end_sample], sr, format='WAV')

def main(args):
    diarization, _, sample_rate = diarize_audio(args.hf_token, args.input_file)
    model = WhisperFacade(model=args.model, quantize=args.quantize)
    model.load_audio(args.input_file)
    #
    writer = WriteSRTIncremental() 
    writer_json = SegmentsJSONWriter()
    whisper_options = {"verbose": None, "word_timestamps": False, 
                       "task": args.task, "suppress_tokens": ""}
    if args.language:
        whisper_options["language"] = args.language
    writer_options = {"max_line_width":55, "max_line_count":2, "word_timestamps": False}
    print("Process diarized blocks")
    
    # Group consecutive segments of the same speaker
    grouped_segments = []
    current_speaker = None
    current_start = None
    current_end = None

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        print(speaker)
        if turn.end - turn.start < 0.5:  # Suppress short utterances (pyannote artifact)
            print(f"start={turn.start:.1f}s stop={turn.end:.1f}s IGNORED")
            continue

        if speaker == current_speaker:
            current_end = turn.end
        else:
            if current_speaker is not None:
                grouped_segments.append((current_start, current_end, current_speaker))
            current_speaker = speaker
            current_start = turn.start
            current_end = turn.end

    # Append the last segment
    if current_speaker is not None:
        grouped_segments.append((current_start, current_end, current_speaker))

    # Process each grouped segment
    for start, end, speaker in grouped_segments:
        clip_path = f"/tmp/speaker_{speaker}_start_{start:.1f}_end_{end:.1f}.wav"
        clip_audio(args.input_file, sample_rate, start, end, clip_path)
        result = model.transcribe(start=start, end=end, options=whisper_options)
        language = result['language']
        print(f"start={start:.1f}s stop={end:.1f}s lang={language} {speaker}")
        writer(result, args.output_file, speaker, start, writer_options)
        writer_json(generate_segments(result['segments'],  speaker, language), args.output_json_file)
    writer_json.finalize()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Diarization and Whisper Transcription CLI")
    parser.add_argument('--model', type=str, required=True, help="Whisper model to use")
    parser.add_argument('--quantize', action='store_true', help="Whether to quantize the model")
    parser.add_argument('--hf-token', type=str, required=True, help="Hugging Face authentication token")
    parser.add_argument('--task', type=str, choices=['transcribe', 'translate'], required=True, help="Task to perform")
    parser.add_argument('--language', type=str, required=False, help="Language to use for transcription or translation")
    parser.add_argument('--input-file', type=str, required=True, help="Input audio file")
    parser.add_argument('--output-file', type=str, required=True, help="Output file for the results (SRT or VTT)")
    parser.add_argument('--output-json-file', type=str, required=True, help="Output file for the results (SRT or VTT)")

    args = parser.parse_args()
    main(args)