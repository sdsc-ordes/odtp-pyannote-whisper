#!/bin/bash

if [ -n "$LANGUAGE" ]; then
    python3 /odtp/odtp-app/app.py \
    --model $MODEL \
    --quantize \
    --hf-token $HF_TOKEN \
    --task $TASK \
    --language $LANGUAGE \
    --input-file /odtp/odtp-input/$INPUT_FILE \
    --output-file /odtp/odtp-output/$OUTPUT_FILE.srt \
    --output-json-file /odtp/odtp-output/$OUTPUT_FILE.json \
    --output-paragraphs-json-file /odtp/odtp-output/$OUTPUT_FILE-paragraphs.json \
    --output-md-file /odtp/odtp-output/$OUTPUT_FILE.md \
    --output-pdf-file /odtp/odtp-output/$OUTPUT_FILE.pdf
else
    python3 /odtp/odtp-app/app.py \
    --model $MODEL \
    --quantize \
    --hf-token $HF_TOKEN \
    --task $TASK \
    --input-file /odtp/odtp-input/$INPUT_FILE \
    --output-file /odtp/odtp-output/$OUTPUT_FILE.srt \
    --output-json-file /odtp/odtp-output/$OUTPUT_FILE.json \
    --output-paragraphs-json-file /odtp/odtp-output/$OUTPUT_FILE-paragraphs.json \
    --output-md-file /odtp/odtp-output/$OUTPUT_FILE.md \
    --output-pdf-file /odtp/odtp-output/$OUTPUT_FILE.pdf
fi
