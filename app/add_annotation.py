import json
import argparse
from datetime import timedelta

def seconds_to_hms(seconds):
    """Convert seconds to HH:MM:SS format."""
    td = timedelta(seconds=seconds)
    return str(td).split(".")[0].zfill(8)  # Ensures HH:MM:SS format

def merge_annotations(annotations_file, host_file, output_file, annotation_type, origin_channel, custom_id):
    # Load the JSON files
    with open(annotations_file, 'r', encoding='utf-8') as ann_file:
        annotations_data = json.load(ann_file)

    with open(host_file, 'r', encoding='utf-8') as host_file:
        host_data = json.load(host_file)

    # Prepare the new annotations
    new_annotations = []
    for segment in annotations_data.get("segments", []):
        new_annotation = {
            "transcript": segment["text"],
            "start_timestamp": seconds_to_hms(segment["start"]),
            "end_timestamp": seconds_to_hms(segment["end"]),
            "labels": {
                "speaker": segment["speaker"],
                "language": segment["language"]
            },
            "tags": []
        }
        new_annotations.append(new_annotation)

    # Use the custom ID
    annotation_entry = {
        "id": custom_id,
        "type": annotation_type,
        "originChannel": origin_channel,
        "labels": {},
        "items": new_annotations
    }

    # Append the new annotation entry to the host file
    host_data.setdefault("annotations", []).append(annotation_entry)

    # Write the updated host file
    with open(output_file, 'w', encoding='utf-8') as out_file:
        json.dump(host_data, out_file, indent=2, ensure_ascii=False)

    print(f"Annotations merged successfully into {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge annotations into a host JSON file.")
    parser.add_argument("annotations_file", help="Path to the annotations JSON file")
    parser.add_argument("host_file", help="Path to the host JSON file")
    parser.add_argument("output_file", help="Path to save the output JSON file")
    parser.add_argument("--type", choices=["audio_transcription", "audio_translation"], required=True,
                        help="Type of annotation (audio_transcription or audio_translation)")
    parser.add_argument("--origin_channel", required=True, help="Origin channel for the annotations")
    parser.add_argument("--id", required=True, help="Custom ID for the new annotation entry")

    args = parser.parse_args()

    merge_annotations(args.annotations_file, args.host_file, args.output_file, args.type, args.origin_channel, args.id)