import json
import os
import argparse
from md2pdf.core import md2pdf

def json_to_markdown(json_data, filename):
    """
    Convert JSON data into a Markdown string.

    :param json_data: List of dictionaries each containing
                      "start", "end", "text", "speaker", "language"
    :param filename:  The name of the JSON file (used as the main title)
    :return: A string of valid Markdown
    """

    # Title
    title = filename.replace("-paragraphs.json", "")
    md_output = f"# {title}\n\n"

    # Transcription Header
    md_output += "## Transcription\n\n"

    # Build the content for each contribution
    for entry in json_data:
        speaker = entry.get("speaker", "Unknown Speaker")
        language = entry.get("language", "Unknown Language")
        start = entry.get("start", "")
        end = entry.get("end", "")
        text = entry.get("text", "")

        # Speaker-language heading
        md_output += f"### {speaker} - {language}\n\n"

        # Table for start/end times
        md_output += "| start       | end         |\n"
        md_output += "|-------------|-------------|\n"
        md_output += f"| {start} | {end} |\n\n"

        # Add the text below the table
        md_output += f"{text}\n\n"

    return md_output


def save_markdown_to_file(markdown_text, output_md_path):
    """
    Save the Markdown text to a file.

    :param markdown_text: The Markdown content as a string.
    :param output_md_path: The path (including filename) where the .md should be saved.
    """
    with open(output_md_path, 'w', encoding='utf-8') as md_file:
        md_file.write(markdown_text)


def markdown_to_pdf(markdown_text, output_pdf_path):
    """
    Convert a Markdown string to a PDF using md2pdf.

    :param markdown_text: The Markdown text to convert.
    :param output_pdf_path: The path (including filename) where the PDF should be saved.
    """
    # Custom CSS (adapt as you like)
    custom_css = r"""
@font-face {
  font-family: "Bitstream Vera Serif Bold";
  src: url("https://mdn.github.io/css-examples/web-fonts/VeraSeBd.ttf");
}

body {
  margin: 0 auto;
  background-color: white;
  font-family: "Bitstream Vera Serif Bold";
  color: #333333;
  line-height: 1;
  max-width: 800px;
  padding: 30px;
  font-size: 12px;
}

p {
  line-height: 150%;
  max-width: 960px;
  font-weight: 400;
  color: #333333;
}

h1,
h2,
h3,
h4 {
  font-weight: 400;
}

h2,
h3,
h4,
h5,
p {
  margin-bottom: 25px;
  padding: 0;
}

h1 {
  margin-bottom: 10px;
  font-size: 300%;
  padding: 0px;
}

h2 {
  font-size: 150%;
}

h3 {
  font-size: 120%;
}

h4 {
  font-size: 100%;
}

h5 {
  font-size: 80%;
  font-weight: 100;
}

h6 {
  font-size: 80%;
  font-weight: 100;
  color: red;
}

a {
  color: grey;
  margin: 0;
  padding: 0;
  vertical-align: baseline;
}

a:hover {
  text-decoration: blink;
  color: green;
}

a:visited {
  color: black;
}

ul,
ol {
  padding: 0;
  margin: 0px 0px 0px 50px;
}

ul {
  list-style-type: square;
  list-style-position: inside;
}

li {
  line-height: 150%;
}

li ul,
li ul {
  margin-left: 24px;
}

pre {
  padding: 0px 24px;
  max-width: 800px;
  white-space: pre-wrap;
}

code {
  font-family: Consolas, Monaco, Andale Mono, monospace;
  line-height: 1.5;
  font-size: 13px;
}

aside {
  display: block;
  float: right;
  width: 390px;
}

blockquote {
  border-left: 0.5em solid #eee;
  padding: 0 1em;
  margin-left: 0;
  max-width: 476px;
}

blockquote cite {
  line-height: 20px;
  color: #bfbfbf;
}

blockquote cite:before {
  content: "\2014 \00A0";
}

blockquote p {
  color: #666;
  max-width: 460px;
}

hr {
  text-align: left;
  margin: 0 auto 0 0;
  color: #999;
}

/* Table styling for the start/end times */
table {
  border-collapse: collapse;
  width: 100%;
  margin-bottom: 15px;
}
table, th, td {
  border: 1px solid #333;
  padding: 6px;
}
th {
  background-color: #eee;
  text-align: left;
}
"""

    # Convert Markdown to PDF directly from a string
    md2pdf(output_pdf_path,
        md_content=markdown_text,  # We pass the markdown text as "raw"
        css_file_path=None,           # If you have an external .css file, you can pass its path here
        base_url=None       # If you have images or relative links
    )

    # If we want the custom CSS from a file, we can write it to a temporary file:
    # Optionally, you could do something like this:
    #
    # with open("custom_style.css", "w", encoding="utf-8") as css_file:
    #     css_file.write(custom_css)
    #
    # md2pdf(output_pdf_path,
    #     raw=markdown_text,
    #     css="custom_style.css",  # pass the CSS file path
    #     extras=[],
    #     base_url=None
    # )


def parse_args():
    """
    Parse command-line arguments using argparse.
    """
    parser = argparse.ArgumentParser(
        description="Convert a JSON-based transcription to Markdown and then to PDF using md2pdf."
    )
    parser.add_argument("input_filename",
                        help="Path to the JSON file containing transcription data.")
    parser.add_argument("--output_md",
                        default="transcription.md",
                        help="Output Markdown file name (default: transcription.md)")
    parser.add_argument("--output_pdf",
                        default="transcription.pdf",
                        help="Output PDF file name (default: transcription.pdf)")
    return parser.parse_args()

def convert_json_to_pdf(input_filename, output_md="transcription.md", output_pdf="transcription.pdf"):
    # 1. Read JSON data
    with open(input_filename, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 2. Convert JSON to Markdown
    markdown_output = json_to_markdown(data, os.path.basename(input_filename))

    # 3. Save Markdown to a .md file (optional)
    save_markdown_to_file(markdown_output, output_md)
    print(f"Markdown saved to: {output_md}")

    # 4. Convert Markdown to PDF using md2pdf
    markdown_to_pdf(markdown_output, output_pdf)
    print(f"PDF generated and saved to: {output_pdf}")

def main():
    args = parse_args()
    convert_json_to_pdf(args.input_filename, args.output_md, args.output_pdf)

if __name__ == '__main__':
    main()