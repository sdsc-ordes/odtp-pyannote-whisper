import gradio as gr
import tempfile
import os
import shutil
import subprocess
import threading
import time
import argparse
import uuid

def create_temp_structure():
    """Create temporary ODTP folder structure"""
    temp_dir = tempfile.mkdtemp(prefix="odtp_")
    os.makedirs(os.path.join(temp_dir, "odtp-input"))
    os.makedirs(os.path.join(temp_dir, "odtp-output"))
    return temp_dir

def remove_later(path, delay):
    time.sleep(delay)
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)

def cleanup_temp(temp_dir):
    """Remove temporary folder structure"""
    shutil.rmtree(temp_dir)

def process_audio(audio_file, 
                  youtube_link, 
                  model, 
                  task, 
                  language, 
                  hf_token=None,
                  quantize=False):
    """Process audio file with Whisper and Pyannote"""
    # Create temp structure
    temp_dir = create_temp_structure()
    
    start_time = time.time()
    print(f"Processing started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
    
    if youtube_link:
        input_path = youtube_link

        id = str(uuid.uuid4())
        output_base = f"{id}"
        output_srt = os.path.join(temp_dir, "odtp-output", f"{id}.srt")
        output_json = os.path.join(temp_dir, "odtp-output", f"{id}.json")
        output_paragraphs_json = os.path.join(temp_dir, "odtp-output", f"{id}_paragraphs.json")
        output_md_file = os.path.join(temp_dir, "odtp-output", f"{id}.md")
        output_pdf_file = os.path.join(temp_dir, "odtp-output", f"{id}.pdf")
    
    else:
        # Copy input file
        input_path = os.path.join(temp_dir, "odtp-input", "input.wav")
        shutil.copy2(audio_file, input_path)

        output_base = audio_file.split("/")[-1].replace(".wav", "")
        output_srt = os.path.join(temp_dir, "odtp-output", f"{output_base}_{task}.srt")
        output_json = os.path.join(temp_dir, "odtp-output", f"{output_base}_{task}.json")
        output_paragraphs_json = os.path.join(temp_dir, "odtp-output", f"{output_base}_{task}_paragraphs.json")
        output_md_file = os.path.join(temp_dir, "odtp-output", f"{output_base}_{task}.md")
        output_pdf_file = os.path.join(temp_dir, "odtp-output", f"{output_base}_{task}.pdf")
    
    # Use HF_TOKEN from environment if not provided
    if not hf_token:
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise ValueError("Hugging Face token is required but not provided.")
    
    # Build command
    cmd = [
        "python3", "/odtp/odtp-app/app.py",
        "--model", model,
        "--hf-token", hf_token,
        "--task", task,
        "--input-file", input_path,
        "--output-file", output_srt,
        "--output-json-file", output_json,
        "--output-paragraphs-json-file", output_paragraphs_json,
        "--output-md-file", output_md_file,
        "--output-pdf-file", output_pdf_file,
        #"--verbose",
    ]

    if quantize:
        cmd.append("--quantize")
    
    if language != "auto":
        cmd.extend(["--language", language])
        
    # Run transcription
    subprocess.run(cmd, check=True)
    
    # Read results
    with open(output_srt, 'r', encoding='utf-8') as f:
        srt_content = f.read()
    with open(output_json, 'r', encoding='utf-8') as f:
        json_content = f.read()

    # Code to delete files after 300 seconds
    threading.Thread(target=remove_later, args=(temp_dir, 300), daemon=True).start()

    end_time = time.time()
    print(f"Processing ended at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")

    total_duration = end_time - start_time
    hours, remainder = divmod(total_duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    total_duration_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
    print(f"Total processing time: {total_duration_str}")

    return total_duration_str, srt_content, json_content, output_srt, output_json, output_paragraphs_json, output_md_file, output_pdf_file

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
                choices=["tiny", 
                         "base", 
                         "small", 
                         "medium", 
                         "large", 
                         "large-v2", 
                         "large-v3", 
                         "large-v3-turbo", 
                         "softcatala/whisper-base-ca", 
                         "projecte-aina/whisper-large-v3-ca-3catparla"],
                value="base",
                label="Whisper Model"
            )
            youtube_link = gr.Textbox(
                type="text",
                label="Youtube Link (optional)",
            )
            task = gr.Dropdown(
                choices=["transcribe", "translate"],
                value="transcribe",
                label="Task"
            )
            language = gr.Dropdown(
                choices=["auto", "en", "es", "ca", "fr", "de", "it", "pt", "nl", "ja", "zh", "ru"],
                value="auto",
                label="Source Language"
            )
            hf_token = gr.Textbox(
                label="Hugging Face Token",
                type="password",
                placeholder="Leave blank if not applicable"
            )
            quantize = gr.Checkbox(
                label="Quantize Model",
                value=False
            )

            submit_btn = gr.Button("Process Audio")
            
        with gr.Column():
            information = gr.Text(
                label="Information"
            )
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
                label="Download SRT File", 
                type="binary"
            )
            json_download = gr.File(
                label="Download JSON File",
                type="binary"
            )
            json_paragraphs_download = gr.File(
                label="Download JSON Paragraphs File",
                type="binary"
            )
            md_download = gr.File(
                label="Download MD File",
                type="binary"
            )
            pdf_download = gr.File(
                label="Download PDF File",
                type="binary"
            )
    
    submit_btn.click(
        fn=process_audio,
        inputs=[audio_input, 
                youtube_link, 
                model, 
                task, 
                language, 
                hf_token,
                quantize],
        outputs=[information, 
                 srt_output, 
                 json_output, 
                 srt_download, 
                 json_download, 
                 json_paragraphs_download,
                 md_download,
                 pdf_download]
    )



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch Gradio app with optional sharing.")
    parser.add_argument('--share', action='store_true', help="Enable sharing the app with a public URL.")
    args = parser.parse_args()

    demo.queue().launch(
        server_name="0.0.0.0",  # More secure default for development
        server_port=7860,         # Default Gradio port
        share=args.share,              # Disable temporary public URL
        show_error=True,          # Show detailed error messages
        debug=True               # Enable debug mode for development
    )
