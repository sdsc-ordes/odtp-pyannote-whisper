---
title: odtp-pyannote-whisper
sdk: docker
pinned: false
---

# odtp-pyannote-whisper

[![Compatible with ODTP v0.5.x](https://img.shields.io/badge/Compatible%20with-ODTP%20v0.5.0-green)]("") [![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-md.svg)](https://huggingface.com/spaces/katospiegel/odtp-pyannote-whisper)

> [!NOTE]  
> This repository makes use of submodules. Therefore, when cloning it you need to include them.
>  
> `git clone --recurse-submodules https://github.com/sdsc-ordes/odtp-pyannote-whisper`

This pipeline processes a `.wav` or `mp4` media file by detecting the number of speakers present in the recording using `pyannote.audio`. For each detected speaker segment, it employs `OpenAI's Whisper model` to transcribe or translate the speech individually. This approach ensures accurate and speaker-specific transcriptions or translations, providing a clear understanding of who said what throughout the audio.

Note: This application utilizes `pyannote.audio` and OpenAI's Whisper model. You must accept the terms of use on Hugging Face for the `pyannote/segmentation` and `pyannote/speaker-diarization` models before using this application.

- [Speaker-Diarization](https://huggingface.co/pyannote/speaker-diarization-3.1)
- [Speaker-Segmentation](https://huggingface.co/pyannote/segmentation-3.0)

After accepting these terms and conditions for those models. You can obtain you HuggingFace API Key to allow the access to these models: 

- [Hugging Face Access Keys](https://huggingface.co/settings/tokens)

This token should be provided to the component via the `ENV` variables or by the corresponding text field in the web app interface ([Here](https://huggingface.com/spaces/katospiegel/odtp-pyannote-whisper)).

![](assets/screenshot.png)


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

| Tool                                            | Version    | Commit Hash | Documentation                                                      |
|-------------------------------------------------|------------|-------------|--------------------------------------------------------------------|
| [OpenAI Whisper](https://github.com/openai/whisper)          | Latest     | [Commit History](https://github.com/openai/whisper/commits/main) | [Whisper Documentation](https://github.com/openai/whisper#readme)  |
| [pyannote.audio](https://github.com/pyannote/pyannote-audio)  | Latest     | [Commit History](https://github.com/pyannote/pyannote-audio/commits/master) | [pyannote.audio Documentation](https://pyannote.github.io/pyannote-audio/) |

## How to add this component to your ODTP instance

This component can be run directly with Docker, however it is designed to be run with [ODTP](https://odtp-org.github.io/odtp-manuals/). In order to add this component to your ODTP CLI, you can use. If you want to use the component directly, please refer to the docker section. 

``` bash
odtp new odtp-component-entry \
--name odtp-pyannote-whisper \
--component-version v0.1.1 \
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

Then create `.env` file similar to `.env.dist` and fill the variables values. Like on this example:

```
MODEL=base
HF_TOKEN=hf_xxxxxxxxxxx
TASK=transcribe
INPUT_FILE=HRC_20220328T0000.mp4
OUTPUT_FILE=HRC_20220328T0000
VERBOSE=TRUE
```

Then create 3 folders: 

- `odtp-input`, where your input data should be located.
- `odtp-output`, where your output data will be stored.
- `odtp-logs`, where the logs will be shared. 

After this, you can run the following command and the pipeline will execute.

``` bash
docker run -it --rm \
-v {PATH_TO_YOUR_INPUT_VOLUME}:/odtp/odtp-input \
-v {PATH_TO_YOUR_OUTPUT_VOLUME}:/odtp/odtp-output \
-v {PATH_TO_YOUR_LOGS_VOLUME}:/odtp/odtp-logs \
--env-file .env \
odtp-pyannote-whisper
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

On Windowss this is the command to execute.

``` powershell
docker run -it --rm `
--gpus all `
-v ${PWD}/odtp-input:/odtp/odtp-input `
-v ${PWD}/odtp-output:/odtp/odtp-output `
-v ${PWD}/odtp-logs:/odtp/odtp-logs `
--env-file .env odtp-pyannote-whisper
```

### Running in API Mode

To run the component in API mode and expose a port, you need to use the following environment variables: 

```
ODTP_API_MODE=TRUE
ODTP_GRADIO_SHARE=FALSE #Only if you want to share the app via the gradio tunneling
```

After the configuration, you can run:

``` bash
docker run -it --rm \
-p 7860:7860 \
--env-file .env \
odtp-pyannote-whisper 
```

And access to the web interface on `localhost:7860` in your browser.

![](assets/screenshot.png)


## Credits and references

This component has been created using the `odtp-component-template` `v0.5.0`. 

The development of this repository has been realized by SDSC.
