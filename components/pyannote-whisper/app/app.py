from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, TextIO, List
import whisper
from whisper import Whisper
import torch
from math import floor, ceil

from transformers import pipeline
import numpy as np
import librosa
import torch
from math import floor, ceil

import os
import argparse
from pyannote.audio import Pipeline, Audio
from whisper.utils import WriteSRT, WriteVTT

import soundfile as sf
import librosa
import json
from dataclasses import dataclass, asdict
from jsonschema import validate, ValidationError

import create_pdf
import paragraphs_creator

from pydub import AudioSegment
import yt_dlp

from slugify import slugify
import uuid
import yaml




class ASRFacade(ABC):
    """Abstract base class to define an interface for transcription."""

    @abstractmethod
    def load_audio(self, file_path: str):
        pass

    @abstractmethod
    def transcribe(
        self,
        start: float,
        end: float,
        options: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Transcribe a portion of audio from start to end time in seconds."""
        pass




class WhisperFacade(ASRFacade):
    wmodel: Whisper
    audio: Any  # The loaded audio array from Whisper

    def __init__(self, model: str, *, quantize=False) -> None:
        print("Initialize Whisper")
        whisper_model = whisper.load_model(model)
        if quantize == True:
            print("Quantize active")
            DTYPE = torch.qint8
            qmodel: Whisper = torch.quantization.quantize_dynamic(
                whisper_model, {torch.nn.Linear}, dtype=DTYPE
            )
            del whisper_model
            self.wmodel = qmodel
        else:
            print("Quantize non active")
            self.wmodel = whisper_model

    def load_audio(self, file_path: str):
        self.audio = whisper.load_audio(file_path)

    def _set_timing_for(self, segment: dict[str, float], offset: float) -> None:
        # Keep this logic the same
        s = segment
        s['start'] += offset 
        s['end']   += offset
        if 'words' in s:
            for w in s['words']:
                w['start'] += offset
                w['end'] += offset

    def transcribe(
        self,
        start: float,
        end: float,
        options: dict[str, Any]
    ) -> dict[str, Any]:
        SAMPLE_RATE = 16_000
        start_index = floor(start * SAMPLE_RATE)
        end_index   = ceil(end * SAMPLE_RATE)

        audio_segment = self.audio[start_index:end_index]
        result = whisper.transcribe(self.wmodel, audio_segment, **options)

        for s in result['segments']:
            self._set_timing_for(segment=s, offset=start)

        return result


class TransformersFacade(ASRFacade):
    """Use a Hugging Face ASR pipeline instead of Whisper."""

    def __init__(self, model_name: str):
        print(f"Initialize Transformers pipeline with model {model_name}")
        self.asr_pipeline = pipeline("automatic-speech-recognition", model=model_name)
        self.sr = 16000
        self.audio_data = None  # We'll store the loaded waveform here

    def load_audio(self, file_path: str):
        # We'll load audio into a single waveform at 16 kHz
        print(f"Loading audio for Transformers from {file_path}")
        waveform, sr = librosa.load(file_path, sr=self.sr, mono=True)
        self.audio_data = waveform
        print(f"Audio loaded: shape={waveform.shape}, sample_rate={sr}")

    def transcribe(
        self,
        start: float,
        end: float,
        options: dict[str, Any]
    ) -> dict[str, Any]:
        """
        We want to return a structure with 'segments' just like Whisper does.
        We'll treat the entire chunk as a single forward pass to the pipeline.
        If you prefer more advanced chunking or word-level timestamps, you can expand this.
        """
        start_sample = floor(start * self.sr)
        end_sample   = ceil(end * self.sr)
        audio_segment = self.audio_data[start_sample:end_sample]

        # The pipeline can handle direct numpy arrays.
        # Some HF pipelines let you pass 'return_timestamps=True' in generate(), but that may vary by model.
        # We'll do a straightforward approach here:
        transcription = self.asr_pipeline(audio_segment)#, sampling_rate=self.sr)

        # We want to unify the output structure with whisper-like dict.
        # Example result for consistency:
        result = {
            "language": "ca",  # or fetch from pipeline if available
            "segments": [
                {
                    "id": 0,
                    "start": start,
                    "end": end,
                    "text": transcription["text"],
                    # If you want words/tokens, you can parse them here if your model supports it
                }
            ]
        }
        return result

def create_asr_facade(model_name: str, quantize: bool = False) -> ASRFacade:
    """Factory function to return either a Whisper or Transformers facade."""
    # Example: if user requests 'base-ca', we switch to Transformers
    if model_name in ['tiny.en', 
                        'tiny', 
                        'base.en', 
                        'base', 
                        'small.en', 
                        'small', 
                        'medium.en', 
                        'medium', 
                        'large-v1', 
                        'large-v2', 
                        'large-v3', 
                        'large', 
                        'large-v3-turbo', 
                        'turbo']:
        return WhisperFacade(model=model_name, quantize=quantize)
    else:
        print(f"Trying to use Transformers model from {model_name}")
        return TransformersFacade(model_name=model_name)
        
    



#############################################################################

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
            # TODO: Add verbose option to print the SRT blocks
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
        self.temp_output_path = ''  # Will store the temporary output file path

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
            self.temp_output_path = os.path.splitext(path)[0] + ".jsonl"
        else:
            if not self.output_path:
                audio_basename = os.path.splitext(os.path.basename(audio_path))[0]
                # Use output_dir if provided, else use the directory of audio_path
                dir = self.output_dir if self.output_dir else os.path.dirname(audio_path)
                self.output_path = os.path.join(dir, audio_basename + ".json")
                self.temp_output_path = os.path.join(dir, audio_basename + ".jsonl")

            path = self.output_path
            mode = 'a' if not self.first_call else 'w'


        if len(segments) > 0 and all(hasattr(segment, 'text') for segment in segments): 
            with open(self.temp_output_path, mode, encoding='utf-8') as f:
                    for idx, segment in enumerate(segments):
                        segment.text = segment.text.strip()
                        json.dump(asdict(segment), f, ensure_ascii=False)
                        f.write('\n')
            self.first_call = False

        # with open(path, mode, encoding='utf-8') as f:
        #     if self.first_call:
        #         # Start the JSON structure
        #         f.write('{"segments": [\n')
        #     else:
        #         f.write(',\n')
        #     for idx, segment in enumerate(segments):
        #         if segment:  # Check if the segment is not empty
        #             if idx > 0:
        #                 f.write(',\n')

        #             segment.text = segment.text.strip()
        #             json.dump(asdict(segment), f, ensure_ascii=False, indent=2)

            self.first_call = False

    def finalize(self):
        """Call this method after all segments have been written to close the JSON array."""
        if self.temp_output_path:
            dict = {"segments": []}
            with open(self.temp_output_path, 'r', encoding='utf-8') as temp_file:
                for line in temp_file:
                    segments = json.loads(line.strip())
                    dict["segments"].append(segments)

        # if self.output_path:
        #     with open(self.output_path, 'a', encoding='utf-8') as f:
        #         f.write('\n  ]\n}\n')

        # This is a quick solution to the error that occurs with empty items in the list ..},,{...
        # However, it's not the best solution. It's better to find and fix the root cause of the issue.
        # One strategy would be read the json file, and append the segments to the list, then write it back.
        # with open(self.output_path, 'r', encoding='utf-8') as file:
        #     content = file.read()
        
        # # Remove extra commas
        # fixed_content = content.replace(',\n,', ',\n')
        # fixed_content = content.replace(',\n,\n,', ',\n')
        # fixed_content = content.replace(',\n,\n,\n,', ',\n')
        
        # # Load the JSON to ensure it's valid
        # try:
        #     json_data = json.loads(fixed_content)
        # except json.JSONDecodeError as e:
        #     raise f"Error decoding JSON: {e}"
        
        # Write the fixed JSON back to the file
        with open(self.output_path, 'w', encoding='utf-8') as file:
            json.dump(dict, file, ensure_ascii=False, indent=2)

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

def convert_mpx_to_wav(file_path, output_path):
    if file_path.lower().endswith('.mp3'):
        # Load the MP3 file
        audio = AudioSegment.from_mp3(file_path)

    elif file_path.lower().endswith('.mp4'):
        audio = AudioSegment.from_file(file_path, format="mp4")

    else:
        raise ValueError("Input file must be an MP3 or MP4 file")
        
    # Define the output path
    #wav_file_path = os.path.splitext(file_path)[0] + '.wav'
    
    # Export as WAV
    audio.export(output_path, format='wav')
    
    return output_path

    
def download_youtube_video(url, filename, output_path='/tmp'):
    ydl_opts = {
        'format': 'bestaudio/best',
        #'outtmpl': os.path.join(output_path, '%(title)s.%(ext)s'),
        'outtmpl': os.path.join(output_path, f"{filename.split('.')[0]}.%(ext)s"),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
    }



    if not os.path.exists(output_path):
        os.makedirs(output_path)

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=True)
        output_file = ydl.prepare_filename(info_dict)
        print(output_file)
        base, ext = os.path.splitext(output_file)
        
        new_file = base + '-original.wav'
        return new_file

import subprocess
import shutil

def convert_video_to_wav(input_file, output_file):
    """
    Convert a .rm file to .wav using FFmpeg.
    
    :param input_file: Path to the input .rm file
    :param output_file: Desired path to the output .wav file
    """
    # Ensure the output directory exists (optional)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Build the FFmpeg command
    # -i input_file  : input file
    # -vn            : ignore video (if any)
    # -acodec pcm_s16le: use 16-bit PCM audio codec
    # -ar 44100      : set sample rate to 44100 Hz
    # -ac 2          : set audio channels to stereo
    command = [
        "ffmpeg",
        "-y",             # overwrite output file if it exists (optional)
        "-i", input_file,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "44100",
        "-ac", "2",
        output_file
    ]
    
    # Run the command
    try:
        subprocess.run(command, check=True)
        print(f"Conversion successful! '{input_file}' has been converted to '{output_file}'.")
    except subprocess.CalledProcessError as e:
        print(f"Error during conversion: {e}")


def main(args):
    # TODO: Take out the file_path from ODTP here
    if args.input_file.startswith('/odtp/odtp-input/http://') or args.input_file.startswith('/odtp/odtp-input/https://'):
        file_path = download_youtube_video(args.input_file.replace("/odtp/odtp-input/",""), filename=os.path.basename(args.output_file) , output_path=os.path.dirname(args.output_file))
        base_slug = slugify(file_path, separator='_')
    elif args.input_file.lower().endswith('.mp3'):
        file_path = convert_mpx_to_wav(args.input_file)
        shutil.copy(file_path, os.path.join("/odtp/odtp-output", os.path.basename(file_path).replace('.wav', '-original.mp3')))
    elif args.input_file.lower().endswith('.wav'):
        file_path = args.input_file
        shutil.copy(file_path, os.path.join("/odtp/odtp-output", os.path.basename(file_path).replace('.wav', '-original.wav')))
    elif args.input_file.lower().endswith('.mp4'):
        file_path = "/odtp/odtp-output/" + os.path.basename(args.input_file).replace('.mp4', '-original.wav')
        convert_mpx_to_wav(args.input_file, file_path)
    elif args.input_file.lower().endswith('.rm'):
        file_path = "/odtp/odtp-output/" + os.path.basename(args.input_file).replace('.rm', '-original.wav')
        convert_video_to_wav(args.input_file, file_path)
    elif args.input_file.lower().endswith('.f4v'):
        file_path = "/odtp/odtp-output/" + os.path.basename(args.input_file).replace('.f4v', '-original.wav')
        convert_video_to_wav(args.input_file, file_path)
    elif args.input_file.lower().endswith('.mkv'):
        file_path = "/odtp/odtp-output/" + os.path.basename(args.input_file).replace('.mkv', '-original.wav')
        convert_video_to_wav(args.input_file, file_path)
    else:
        raise ValueError(f"Input file must be an MP3, WAV, RM, F4V, MKV, Youtube Link, or MP4 file. Input file: {args.input_file}")
    
    
    diarization, _, sample_rate = diarize_audio(args.hf_token, file_path)

    # Create the correct ASR facade
    asr_model = create_asr_facade(args.model, quantize=args.quantize)
    asr_model.load_audio(file_path)
    
    writer = WriteSRTIncremental() 
    writer_json = SegmentsJSONWriter()
    
    # Whisper-like transcription options
    whisper_options = {
        "verbose": None,
        "word_timestamps": False,
        "task": args.task,
        "suppress_tokens": ""
    }
    if args.language:
        # This is only relevant for Whisper. For Transformers,
        # you might specify a different approach or ignore it.
        whisper_options["language"] = args.language

    writer_options = {
        "max_line_width": 55,
        "max_line_count": 2,
        "word_timestamps": False
    }

    if args.verbose:
        print("Process diarized blocks")

    
    grouped_segments = []
    current_speaker = None
    current_start   = None
    current_end     = None

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        if args.verbose:
            print(speaker)
        if turn.end - turn.start < 0.5:
            # ignore short utterances
            continue
        if speaker == current_speaker:
            current_end = turn.end
        else:
            if current_speaker is not None:
                grouped_segments.append((current_start, current_end, current_speaker))
            current_speaker = speaker
            current_start    = turn.start
            current_end      = turn.end

    if current_speaker is not None:
        grouped_segments.append((current_start, current_end, current_speaker))

    # Process each grouped segment
    for start, end, speaker in grouped_segments:
        clip_path = f"/tmp/speaker_{speaker}_start_{start:.1f}_end_{end:.1f}.wav"
        clip_audio(file_path, sample_rate, start, end, clip_path)
        
        # Important: we call asr_model instead of model
        result = asr_model.transcribe(start=start, end=end, options=whisper_options)
        language = result.get('language', args.language or 'unknown')
        
        if args.verbose:
            print(f"start={start:.1f}s stop={end:.1f}s lang={language} {speaker}")

        # Use your existing logic to write SRT, JSON, etc.
        writer(result, args.output_file, speaker, start, writer_options)
        writer_json(generate_segments(result['segments'], speaker, language), args.output_json_file)

    writer_json.finalize()

    # If you want to validate JSON, paragraphs, PDF creation, etc.
    paragraphs_creator.process_paragraphs(
        args.output_json_file,
        args.output_paragraphs_json_file,
        3
    )
    create_pdf.convert_json_to_pdf(
        args.output_paragraphs_json_file,
        args.output_md_file,
        args.output_pdf_file
    )



if __name__ == '__main__':
    # Multiprocessing requires spawn when working with CUDA
    #multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser(description="Diarization and Whisper Transcription CLI")
    parser.add_argument('--model', type=str, required=True, help="Whisper model to use")
    parser.add_argument('--quantize', action='store_true', help="Whether to quantize the model")
    parser.add_argument('--hf-token', type=str, required=True, help="Hugging Face authentication token")
    parser.add_argument('--task', type=str, choices=['transcribe', 'translate'], required=True, help="Task to perform")
    parser.add_argument('--language', type=str, required=False, help="Language to use for transcription or translation")
    parser.add_argument('--input-file', type=str, required=True, help="Input audio file")
    parser.add_argument('--output-file', type=str, required=True, help="Output file for the results (SRT or VTT)")
    parser.add_argument('--output-json-file', type=str, required=True, help="Output json file.")
    parser.add_argument('--output-paragraphs-json-file', type=str, required=True, help="Output paragraphs file")
    parser.add_argument('--output-md-file', type=str, required=True, help="Output markdown file")
    parser.add_argument('--output-pdf-file', type=str, required=True, help="Output pdf file")
    parser.add_argument('--verbose', action='store_true', help="Printing status")

    args = parser.parse_args()
    main(args)