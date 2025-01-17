import json
import sys

def seconds_to_hhmmss(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"


def create_paragraphs(annotations, min_gap=4):
    paragraphs = []
    current_paragraph = {}
    prev_end_time = None
    prev_speaker = None

    for item in annotations:
        start_time = item["start"]
        end_time = item["end"]
        transcript = item["text"]
        speaker = item["speaker"]
        language = item["language"]

        # Start a new paragraph if timestamp gap exceeds min_gap or speaker changes
        if (
            prev_end_time is not None
            and (
                start_time - prev_end_time > min_gap
                or speaker != prev_speaker
                or (current_paragraph and language != current_paragraph["language"])
            )
        ):
            paragraphs.append(current_paragraph)
            current_paragraph = []

        if current_paragraph:
            current_paragraph["text"] += " " + transcript
            current_paragraph["end"] = seconds_to_hhmmss(end_time)
        else:
            current_paragraph = {
                "start": seconds_to_hhmmss(start_time),
                "end": seconds_to_hhmmss(end_time),
                "text": transcript,
                "speaker": speaker,
                "language": language
            }

        prev_end_time = end_time
        prev_speaker = speaker

    if current_paragraph:
        paragraphs.append(current_paragraph)

    return paragraphs

def process_paragraphs(input_file, output_file, min_gap):
    with open(input_file, 'r') as f:
        json_data = json.load(f)

    annotations = json_data["segments"]

    result = create_paragraphs(annotations, min_gap)

    with open(output_file, 'w') as f:
        json.dump(result, f, indent=4)

def main():
    if len(sys.argv) != 4:
        print("Usage: python paragraphsCreator.py <input_json_file> <output_json_file> <min_gap>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    min_gap = int(sys.argv[3])

    process_paragraphs(input_file, output_file, min_gap)

if __name__ == "__main__":
    main()