FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Set environment variable to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Weasyprint is necessary for pdf printing
RUN apt-get update && apt-get install -y apt-utils weasyprint 

RUN apt-get install -y python3.11 python3.11-venv python3-pip

COPY odtp-component-client/requirements.txt /tmp/odtp.requirements.txt
RUN pip install -r /tmp/odtp.requirements.txt


#######################################################################
# PLEASE INSTALL HERE ALL SYSTEM DEPENDENCIES RELATED TO YOUR TOOL
#######################################################################

# Installing dependecies from the app
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Dependencies

RUN apt-get update && \
    apt-get install -y zip git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# ffmpeg
COPY --link --from=mwader/static-ffmpeg:6.1.1 /ffmpeg /usr/local/bin/
COPY --link --from=mwader/static-ffmpeg:6.1.1 /ffprobe /usr/local/bin/


######################################################################
# ODTP COMPONENT CONFIGURATION. 
# DO NOT TOUCH UNLESS YOU KNOW WHAT YOU ARE DOING.
######################################################################

##################################################
# ODTP Preparation
##################################################

RUN mkdir /odtp \
    /odtp/odtp-config \
    /odtp/odtp-app \
    /odtp/odtp-component-client \
    /odtp/odtp-logs \ 
    /odtp/odtp-input \
    /odtp/odtp-workdir \
    /odtp/odtp-output 

# This last 2 folders are specific from odtp-eqasim
RUN mkdir /odtp/odtp-workdir/cache \
    /odtp/odtp-workdir/output 

# This copy all the information for running the ODTP component
COPY odtp.yml /odtp/odtp-config/odtp.yml

COPY ./odtp-component-client /odtp/odtp-component-client

COPY ./app /odtp/odtp-app
WORKDIR /odtp

##################################################
# Fix for end of the line issue on Windows
##################################################

# Fix for end of the line issue on Windows. Avoid error when building on windows
RUN find /odtp -type f -iname "*.sh" -exec sed -i 's/\r$//' {} \;


ENTRYPOINT ["bash", "/odtp/odtp-component-client/startup.sh"]