#!/bin/bash

############################################################################################
# START OF MANUAL CONFIGURATION. 
# ADAPT THE TEMPLATE HERE.
############################################################################################

#########################################################
# 1. GITHUB CLONING OF REPO
# Clone the repository of your tool and checkout to one specific commit. 
#########################################################

# git clone https://github.com/odtp-org/tool-example.git /odtp/odtp-workdir/tool-example
# cd /odtp/odtp-workdir/tool-example
# git checkout xxxxxxxxxxxx

#########################################################
# 2. CONFIG FILE CONFIGURATION
# Read placeholders and create config file from Environment  
#########################################################

# python3 /odtp/odtp-component-client/parameters.py /odtp/odtp-app/config_templates/template.yml /odtp/odtp-workdir/config.yml

#########################################################
# 3. INPUT FOLDER MANAGEMENT
#########################################################

# ln -s /odtp/odtp-input/... /odtp/odtp-workdir/...

#########################################################
# 4. TOOL EXECUTION
# While the output is managed by ODTP and placed in /odtp/odtp-output/
#########################################################

if [ -n "$LANGUAGE" ]; then
    python3 /odtp/odtp-app/app.py --model $MODEL --quantize --hf-token $HF_TOKEN --task $TASK --language $LANGUAGE --input-file /odtp/odtp-input/$INPUT_FILE --output-file /odtp/odtp-output/$OUTPUT_FILE.translate.srt --output-json-file /odtp/odtp-output/$OUTPUT_FILE.translate.json
else
    python3 /odtp/odtp-app/app.py --model $MODEL --quantize --hf-token $HF_TOKEN --task $TASK --input-file /odtp/odtp-input/$INPUT_FILE --output-file /odtp/odtp-output/$OUTPUT_FILE.srt --output-json-file /odtp/odtp-output/$OUTPUT_FILE.json 
fi

#########################################################
# 5. OUTPUT FOLDER MANAGEMENT
# The selected output files generated should be placed in the output folder
#########################################################

# cp -r /odtp/odtp-workdir/output/* /odtp/odtp-output

############################################################################################
# END OF MANUAL USER APP
############################################################################################
