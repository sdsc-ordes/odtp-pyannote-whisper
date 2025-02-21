#!/bin/bash

if [ -n "$FULL_PIPELINE" ]; then

    echo "RUNNING TRANSCRIPTION AND EN TRANSLATION PIPELINE" 
    python3 /odtp/odtp-app/app.py \
    --model $MODEL \
    $( [ "$QUANTIZE" = "TRUE" ] && echo "--quantize" ) \
    --hf-token $HF_TOKEN \
    --task transcribe \
    --input-file /odtp/odtp-input/$INPUT_FILE \
    --output-file /odtp/odtp-output/$OUTPUT_FILE-transcription_original.srt \
    --output-json-file /odtp/odtp-output/$OUTPUT_FILE-transcription_original.json \
    --output-paragraphs-json-file /odtp/odtp-output/${OUTPUT_FILE}-transcription_original_paragraphs.json \
    --output-md-file /odtp/odtp-output/$OUTPUT_FILE-transcription_original_original.md \
    --output-pdf-file /odtp/odtp-output/$OUTPUT_FILE-transcription_original_original.pdf \
    $( [ "$VERBOSE" = "TRUE" ] && echo "--verbose" )

    python3 /odtp/odtp-app/app.py \
    --model $MODEL \
    $( [ "$QUANTIZE" = "TRUE" ] && echo "--quantize" ) \
    --hf-token $HF_TOKEN \
    --task translate \
    --language en \
    --input-file /odtp/odtp-input/$INPUT_FILE \
    --output-file /odtp/odtp-output/$OUTPUT_FILE-translation_original_english.srt \
    --output-json-file /odtp/odtp-output/$OUTPUT_FILE-translation_original_english.json \
    --output-paragraphs-json-file /odtp/odtp-output/${OUTPUT_FILE}-translation_original_english_paragraphs.json \
    --output-md-file /odtp/odtp-output/$OUTPUT_FILE-translation_original_english.md \
    --output-pdf-file /odtp/odtp-output/$OUTPUT_FILE-translation_original_english.pdf \
    $( [ "$VERBOSE" = "TRUE" ] && echo "--verbose" )

    echo "Adding annotations"
    python3 /odtp/odtp-app/add_annotation.py \
    /odtp/odtp-output/$OUTPUT_FILE-transcription_original.json \
    /odtp/odtp-input/$INPUT_METADATA_FILE \
    /odtp/odtp-output/$OUTPUT_FILE.json \
    --type audio_transcription \
    --origin_channel original \
    --id transcription_original

    python3 /odtp/odtp-app/add_annotation.py \
    /odtp/odtp-output/$OUTPUT_FILE-translation_original_english.json \
    /odtp/odtp-output/$OUTPUT_FILE.json \
    /odtp/odtp-output/$OUTPUT_FILE.json \
    --type audio_translation \
    --origin_channel original \
    --id translation_original_english

    echo "Generating yml file"
    python3 /odtp/odtp-app/project_metadata_export.py /odtp/odtp-output/

    echo "Uploading to S3"
    #python3 /odtp/odtp-app/s3_upload.py
    #TBD

else
    python3 /odtp/odtp-app/app.py \
    --model $MODEL \
    $( [ "$QUANTIZE" = "TRUE" ] && echo "--quantize" ) \
    --hf-token $HF_TOKEN \
    --task $TASK \
    $( [ -n "$LANGUAGE" ] && echo "--language $LANGUAGE" ) \
    --input-file /odtp/odtp-input/$INPUT_FILE \
    --output-file /odtp/odtp-output/$OUTPUT_FILE.srt \
    --output-json-file /odtp/odtp-output/$OUTPUT_FILE.json \
    --output-paragraphs-json-file /odtp/odtp-output/${OUTPUT_FILE}_paragraphs.json \
    --output-md-file /odtp/odtp-output/$OUTPUT_FILE.md \
    --output-pdf-file /odtp/odtp-output/$OUTPUT_FILE.pdf \
    $( [ "$VERBOSE" = "TRUE" ] && echo "--verbose" )
fi

