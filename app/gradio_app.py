import gradio as gr
import tempfile
import os
import shutil
import subprocess
from pathlib import Path
import io

def create_temp_structure():
    """Create temporary ODTP folder structure"""
    temp_dir = tempfile.mkdtemp(prefix="odtp_")
    os.makedirs(os.path.join(temp_dir, "odtp-input"))
    os.makedirs(os.path.join(temp_dir, "odtp-output"))
    return temp_dir

def cleanup_temp(temp_dir):
    """Remove temporary folder structure"""
    shutil.rmtree(temp_dir)

def process_audio(audio_file, model, task, language, hf_token):
    """Process audio file with Whisper and Pyannote"""
    # Create temp structure
    temp_dir = create_temp_structure()
    
    try:
        # Copy input file
        input_path = os.path.join(temp_dir, "odtp-input", "input.wav")
        shutil.copy2(audio_file, input_path)
        
        # Prepare output paths
        output_base = "output"
        output_srt = os.path.join(temp_dir, "odtp-output", 
            f"{output_base}.{'translate.' if task == 'translate' else ''}srt")
        output_json = os.path.join(temp_dir, "odtp-output",
            f"{output_base}.{'translate.' if task == 'translate' else ''}json")
        
        # Build command
        cmd = [
            "python3", "/odtp/odtp-app/app.py",
            "--model", model,
            "--quantize",
            "--hf-token", hf_token,
            "--task", task,
            "--input-file", input_path,
            "--output-file", output_srt,
            "--output-json-file", output_json
        ]
        
        if language != "auto":
            cmd.extend(["--language", language])
            
        # Run transcription
        subprocess.run(cmd, check=True)
        
        # Read results
        with open(output_srt, 'r', encoding='utf-8') as f:
            srt_content = f.read()
        with open(output_json, 'r', encoding='utf-8') as f:
            json_content = f.read()
        
        # Create BytesIO objects for downloads
        srt_bytes = io.BytesIO(srt_content.encode('utf-8'))
        srt_bytes.name = "output.srt"
        json_bytes = io.BytesIO(json_content.encode('utf-8'))
        json_bytes.name = "output.json"
        
        # Return contents and BytesIO objects
        return srt_content, json_content, srt_bytes, json_bytes
        
    finally:
        # Cleanup
        cleanup_temp(temp_dir)

# Define Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Audio Transcription/Translation with Speaker Diarization")
    
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(
                type="filepath",
                label="Upload Audio File (WAV format)"
            )
            model = gr.Dropdown(
                choices=["tiny", "base", "small", "medium", "large", "large-v2"],
                value="base",
                label="Whisper Model"
            )
            task = gr.Dropdown(
                choices=["transcribe", "translate"],
                value="transcribe",
                label="Task"
            )
            language = gr.Dropdown(
                choices=["auto", "en", "es", "fr", "de", "it", "pt", "nl", "ja", "zh", "ru"],
                value="auto",
                label="Source Language"
            )
            hf_token = gr.Textbox(
                label="Hugging Face Token",
                type="password"
            )
            submit_btn = gr.Button("Process Audio")
            
        with gr.Column():
            srt_output = gr.Textbox(
                label="SRT Output",
                lines=10
            )
            json_output = gr.Textbox(
                label="JSON Output",
                lines=10
            )
            # Add download buttons
            srt_download = gr.File(
                label="Download SRT File"
            )
            json_download = gr.File(
                label="Download JSON File"
            )
    
    submit_btn.click(
        fn=process_audio,
        inputs=[audio_input, model, task, language, hf_token],
        outputs=[srt_output, json_output, srt_download, json_download]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",  # More secure default for development
        server_port=7860,         # Default Gradio port
        share=False,              # Disable temporary public URL
        show_error=True,          # Show detailed error messages
        debug=True               # Enable debug mode for development
    )