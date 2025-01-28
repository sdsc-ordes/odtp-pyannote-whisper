# odtp-pyannote-whisper

This component is still under development. 

Add here your badges:
[![Launch in your ODTP](https://img.shields.io/badge/Launch%20in%20your-ODTP-blue?logo=launch)](http://localhost:8501/launch-component)
[![Compatible with ODTP v0.5.x](https://img.shields.io/badge/Compatible%20with-ODTP%20v0.5.0-green)]("")

> [!NOTE]  
> This repository makes use of submodules. Therefore, when cloning it you need to include them.
>  
> `git clone --recurse-submodules https://github.com/sdsc-ordes/odtp-pyannote-whisper`

This pipeline processes a `.wav` audio file by detecting the number of speakers present in the recording using `pyannote.audio`. For each detected speaker segment, it employs `OpenAI's Whisper model` to transcribe or translate the speech individually. This approach ensures accurate and speaker-specific transcriptions or translations, providing a clear understanding of who said what throughout the audio.

Note: This application utilizes `pyannote.audio` and OpenAI's Whisper model. You must accept the terms of use on Hugging Face for the `pyannote/segmentation` and `pyannote/speaker-diarization` models before using this application.

## Table of Contents

- [Tools Information](#tools-information)
- [How to add this component to your ODTP instance](#how-to-add-this-component-to-your-odtp-instance)
- [Data sheet](#data-sheet)
    - [Parameters](#parameters)
    - [Secrets](#secrets)
    - [Input Files](#input-files)
    - [Output Files](#output-files)
- [Tutorial](#tutorial)
    - [How to run this component as docker](#how-to-run-this-component-as-docker)
    - [Development Mode](#development-mode)
    - [Running with GPU](#running-with-gpu)
    - [Running in API Mode](#running-in-api-mode)
- [Credits and References](#credits-and-references)

## Tools Information

| Tool | Semantic Versioning | Commit | Documentation |
| --- | --- | --- | --- |
| Tool                                            | Version    | Commit Hash | Documentation                                                      |
|-------------------------------------------------|------------|-------------|--------------------------------------------------------------------|
| [OpenAI Whisper](https://github.com/openai/whisper)          | Latest     | [Commit History](https://github.com/openai/whisper/commits/main) | [Whisper Documentation](https://github.com/openai/whisper#readme)  |
| [pyannote.audio](https://github.com/pyannote/pyannote-audio)  | Latest     | [Commit History](https://github.com/pyannote/pyannote-audio/commits/master) | [pyannote.audio Documentation](https://pyannote.github.io/pyannote-audio/) |

## How to add this component to your ODTP instance

In order to add this component to your ODTP CLI, you can use. If you want to use the component directly, please refer to the docker section. 

``` bash
odtp new odtp-component-entry \
--name odtp-pyannote-whisper \
--component-version v0.0.1 \
--repository https://github.com/sdsc-ordes/odtp-pyannote-whisper 
```

## Data sheet

### Parameters

| Parameter    | Description                                            | Type   | Required | Default Value | Possible Values                                                   | Constraints                        |
|--------------|--------------------------------------------------------|--------|----------|---------------|-------------------------------------------------------------------|------------------------------------|
| `MODEL`      | Whisper model to use for transcription or translation  | String | Yes      | `large-v3`    | `tiny`, `base`, `small`, `medium`, `large`, `large-v2`, `large-v3` | Must be a valid Whisper model name |
| `TASK`       | Task to perform on the audio input                     | String | Yes      | `transcribe`  | `transcribe`, `translate`                                         | Must be `transcribe` or `translate` |
| `LANGUAGE`   | Source language code for the audio input               | String | No       | `auto`        | `auto`, `en`, `es`, `fr`, `de`, `it`, `pt`, `nl`, `ja`, `zh`, `ru` | Must be a supported language code  |
| `INPUT_FILE` | Path to the input `.wav` audio file                    | String | Yes      | N/A           | Any valid file path to a `.wav` file                              | File must exist and be accessible  |
| `OUTPUT_FILE`| Base name for the output files (without extension)     | String | Yes      | `output`      | Any valid file name                                               | Should not contain invalid characters |

### Secrets

| Secret Name | Description                             | Type   | Required | Default Value | Constraints    | Notes                                                         |
|-------------|-----------------------------------------|--------|----------|---------------|----------------|---------------------------------------------------------------|
| HF_TOKEN    | Hugging Face API token for model access | String | Yes      | None          | Valid API Token | Obtain from your Hugging Face account settings   |

### Input Files

| File/Folder     | Description                       | File Type | Required | Format      | Notes                                            |
|-----------------|-----------------------------------|-----------|----------|-------------|--------------------------------------------------|
| `INPUT_FILE`    | Input audio file for processing   | `.wav`    | Yes      | WAV format  | Path specified by `INPUT_FILE` parameter         |

### Output Files

| File/Folder          | Description                          | File Type | Contents                     | Usage                                            |
|----------------------|--------------------------------------|-----------|------------------------------|--------------------------------------------------|
| `OUTPUT_FILE.srt`    | Transcribed subtitles in SRT format  | `.srt`    | Transcribed text with timings | Use with video players to display subtitles      |
| `OUTPUT_FILE.json`   | Transcription data in JSON format    | `.json`   | Detailed transcription data   | For programmatic access and data analysis        |

## Tutorial

### How to run this component as docker

Build the dockerfile.

``` bash
docker build -t odtp-pyannote-whisper .
```

Run the following command. Mount the correct volumes for input/output/logs folders.

``` bash
docker run -it --rm \
-v {PATH_TO_YOUR_INPUT_VOLUME}:/odtp/odtp-input \
-v {PATH_TO_YOUR_OUTPUT_VOLUME}:/odtp/odtp-output \
-v {PATH_TO_YOUR_LOGS_VOLUME}:/odtp/odtp-logs \
--env-file .env odtp-pyannote-whisper
```

### Development Mode

To run the component in development mode, mount the app folder inside the container:

``` bash
docker run -it --rm \
-v {PATH_TO_YOUR_INPUT_VOLUME}:/odtp/odtp-input \
-v {PATH_TO_YOUR_OUTPUT_VOLUME}:/odtp/odtp-output \
-v {PATH_TO_YOUR_LOGS_VOLUME}:/odtp/odtp-logs \
-v {PATH_TO_YOUR_APP_FOLDER}:/odtp/app \
--env-file .env odtp-pyannote-whisper
```

### Running with GPU

To run the component with GPU support, use the following command:

``` bash
docker run -it --rm \
--gpus all \
-v {PATH_TO_YOUR_INPUT_VOLUME}:/odtp/odtp-input \
-v {PATH_TO_YOUR_OUTPUT_VOLUME}:/odtp/odtp-output \
-v {PATH_TO_YOUR_LOGS_VOLUME}:/odtp/odtp-logs \
--env-file .env odtp-pyannote-whisper
```

### Running in API Mode

To run the component in API mode and expose a port, use the following command:

``` bash
docker run -it --rm \
-v {PATH_TO_YOUR_INPUT_VOLUME}:/odtp/odtp-input \
-v {PATH_TO_YOUR_OUTPUT_VOLUME}:/odtp/odtp-output \
-v {PATH_TO_YOUR_LOGS_VOLUME}:/odtp/odtp-logs \
-p {HOST_PORT}:7860 \
--env-file .env \
--entrypoing python3 \
odtp-pyannote-whisper \
/odtp/odtp-app/gradio_app.py
```

## Credits and references

SDSC

This component has been created using the `odtp-component-template` `v0.5.0`. 
