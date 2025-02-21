#!/usr/bin/env python3
import argparse
import os
import re
import yaml

def parse_basename_and_date(folder):
    """
    Searches the folder for a file matching the pattern 'HRC_YYYYMMDDT[HHMM]'.
    Returns the base name and a formatted session date (e.g., "2016 06 22 00:00").
    """
    pattern = re.compile(r"^(HRC_\d{8}T\d{4})")
    for filename in os.listdir(folder):
        match = pattern.match(filename)
        if match:
            base_name = match.group(1)
            # Extract date and time parts from the base name
            dt_match = re.match(r"HRC_(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})", base_name)
            if dt_match:
                year, month, day, hour, minute = dt_match.groups()
                session_date = f"{year} {month} {day} {hour}:{minute}"
                return base_name, session_date
    return None, None

def check_video_file(folder, base_name):
    """
    Checks if an MP4 file with the given base name exists in the folder.
    """
    video_filename = f"{base_name}.mp4"
    return video_filename in os.listdir(folder)

def generate_metadata(base_name, session_date, include_video):
    """
    Builds a metadata dictionary containing file entries based on the base name,
    session date, and whether a video file is present.
    """
    metadata = {
        "files": [
            {
                "name": f"{base_name}.json",
                "type": "json",
                "description": f"JSON file containing metadata transcription and translation from the {session_date} session"
            },
            {
                "name": f"{base_name}-files.yml",
                "type": "yml",
                "description": f"YAML file containing metadata of the files from the {session_date} session"
            }
        ]
    }
    
    if include_video:
        metadata["files"].append({
            "name": f"{base_name}.mp4",
            "type": "mp4",
            "description": f"MP4 video file from the {session_date} session"
        })
    
    metadata["files"].extend([
        {
            "name": f"{base_name}-original.wav",
            "type": "wav",
            "description": f"Original audio file from the {session_date} session"
        },
        {
            "name": f"{base_name}-transcription_original.srt",
            "type": "srt",
            "description": f"Transcription file in SRT format from the original audio of the {session_date} session"
        },
        {
            "name": f"{base_name}-transcription_original.pdf",
            "type": "pdf",
            "description": f"PDF file containing the transcription from the original audio of the {session_date} session"
        },
        {
            "name": f"{base_name}-translation_original_english.srt",
            "type": "srt",
            "description": f"Translation file in SRT format to English from the original audio of the {session_date} session"
        },
        {
            "name": f"{base_name}-translation_original_english.pdf",
            "type": "pdf",
            "description": f"PDF file containing the English translation from the original audio of the {session_date} session"
        }
    ])
    
    return metadata

def write_yaml_file(metadata, output_file):
    """
    Writes the metadata dictionary to a YAML file.
    """
    with open(output_file, "w") as f:
        yaml.dump(metadata, f, sort_keys=False, default_flow_style=True)
    print(f"Metadata YAML file written to {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Generate YAML metadata for session files in a folder."
    )
    parser.add_argument("folder", help="Path to the folder containing the session files.")
    args = parser.parse_args()
    
    folder = args.folder
    if not os.path.isdir(folder):
        print(f"Error: {folder} is not a valid directory.")
        return
    
    base_name, session_date = parse_basename_and_date(folder)
    if not base_name:
        print("Error: Could not find a file matching the expected pattern 'HRC_YYYYMMDDT[HHMM]' in the folder.")
        return
    
    include_video = check_video_file(folder, base_name)
    metadata = generate_metadata(base_name, session_date, include_video)
    
    # Output file is always in the same folder and named as <base_name>-files.yml
    output_file = os.path.join(folder, f"{base_name}-files.yml")
    write_yaml_file(metadata, output_file)

if __name__ == "__main__":
    main()
