#!/bin/bash

if [ "${ODTP_GRADIO_SHARE:-FALSE}" == "TRUE" ]; then
    python3 /odtp/odtp-app/gradio_app.py --share
else
    python3 /odtp/odtp-app/gradio_app.py
fi