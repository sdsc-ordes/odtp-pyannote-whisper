FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

RUN apt-get update && apt-get install -y apt-utils

RUN apt-get install -y python3.11 python3.11-venv python3-pip

# Create directories and set permissions before switching to the non-root user
RUN mkdir -p /odtp/odtp-tmp \
    /odtp \
    /odtp/odtp-config \
    /odtp/odtp-app \
    /odtp/odtp-component-client \
    /odtp/odtp-logs \
    /odtp/odtp-input \
    /odtp/odtp-workdir \
    /odtp/odtp-output \
    /home/user && \
    chown -R 1000:1000 /odtp /home/user


COPY odtp-component-client/requirements.txt /odtp/odtp-tmp/odtp.requirements.txt
RUN pip install -r /odtp/odtp-tmp/odtp.requirements.txt

#######################################################################
# PLEASE INSTALL HERE ALL SYSTEM DEPENDENCIES RELATED TO YOUR TOOL
#######################################################################

# Installing dependecies from the app
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Dependencies

RUN apt-get update && \
    apt-get install -y zip git libglib2.0-0 libpango1.0-0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# ffmpeg
COPY --link --from=mwader/static-ffmpeg:6.1.1 /ffmpeg /usr/local/bin/
COPY --link --from=mwader/static-ffmpeg:6.1.1 /ffprobe /usr/local/bin/

# Adjust permissions so user 1000 can access /usr/local/bin
RUN chown -R 1000:1000 /usr/local/bin/

######################################################################
# ODTP COMPONENT CONFIGURATION. 
# DO NOT TOUCH UNLESS YOU KNOW WHAT YOU ARE DOING.
######################################################################

##################################################
# ODTP Preparation
##################################################

# Switch to the "user" user
USER 1000

# Set home to the user's home directory
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

# This copy all the information for running the ODTP component
COPY odtp.yml /odtp/odtp-config/odtp.yml

COPY ./odtp-component-client /odtp/odtp-component-client

COPY ./app /odtp/odtp-app
WORKDIR /odtp

##################################################
# Fix for end of the line issue on Windows
##################################################

# Switch back to root user to run sed command
USER root
RUN chown -R 1000:1000 /odtp

# Switch back to the "user" user
USER 1000
# Fix for end of the line issue on Windows. Avoid error when building on windows
RUN find /odtp -type f -iname "*.sh" -exec sed -i 's/\r$//' {} \;

EXPOSE 7860

ENTRYPOINT ["bash", "/odtp/odtp-component-client/startup.sh"]