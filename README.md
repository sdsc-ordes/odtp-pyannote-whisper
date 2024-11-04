# ODTP Component Template

This is a template that facilitates the development of new `odtp-components`. An `odtp` compatible component is a docker container able to perform a functional unit of computing in the digital twin. You can think of it as a blackbox that takes inputs files and/or parameters and perfom a task. Usually this lead to some files as a result (Ephemeral component), or to a visualization (Interactive component).

Internally a component will run a bash script `./app/app.sh` that must include the commands for running your tool, and managing the input/output logic. While input files are located in the folder `/odtp/odtp-input`, parameters values are represented by environment variables within the component. In this way you can access to them by using `$` before the name of your variable. Finally, the output files generated are requested to be placed in `/odtp/odtp-output/`.

## How to clone this repository? 

> [!NOTE]  
> This repository makes use of submodules. Therefore, when cloning it you need to include them.
>  
> `git clone --recurse-submodules https://github.com/odtp-org/odtp-component-template`


## How to create an odtp compatible component using this template?

1. Identify which parameters would you like to expose.
2. Configure the Dockerfile to install all the OS requirements needed for your tool to run. 
    1. (Optional) If your tool requires python, and the dependencies offered in the repo are not compatible with the docker image you can configure custom dependencies in requirements.txt
3. Configure the `app/app.sh` file to:
    1. Clone the repository of your tool and checkout to one specific commit. 
    2. (Optional) If your app uses a config file (i.e. `config.yml` or `config.json`), you need to provide a templace including placeholders for the variables you would like to expose. Placeholders can be defined by using double curly braces wrapping the name of the variable, such as `{{VARIABLE}}`. Then you can run `python3 /odtp/odtp-component-client/parameters.py PATH_TO_TEMPLATE PATH_TO_OUTPUT_CONFIG_FILE` and every placeholder will be replaced by the value in the environment variable.
    3. Copy (`cp -r`) or create symbolic links (`ln -s`) to locate the input files in `/odpt/odtp-input/` in the folder. 
    4. Run the tool. You can access to the parameters as environemnt variables (i.e. `$PARAMETER_A`)
    5. Manage the output exporting. At the end of the component execution all generated output should be located in `/odtp/odtp-output`. Copy all output files into this folder. 
4. Describe all the metadata in `odtp.yml`. Please check below for instructions.
5. Publish your tool in the ODTP Zoo. (Temporaly unavailable)

### Semantic Validation

ODTP will be able to validate the input/output files. In order to do this we use SHACL validation. However, the developer should provide a schema of the input/output schema. This section is still under development and it will be available soon. 

## Internal data structure of a component

It's important to remark that when the container is built an specific folder structure is generated:

- `/odtp`: The main folder.
- `/odtp/odtp-component-client`: This is the odtp client that will manage the execution, logging, and input/output functions of the component. It is include as a submodule, and the user doesn't need to modify it.
- `/odtp/odtp-app`: This folder have the content of `/app` folder in this template. It contains the tool execution bash script and the tool configuration files. 
- `/odtp/odtp-workdir`: This is the working directory where the tool repository should be placed and all the middle files such as cache folders.
- `/odtp/odtp-input`: Input folder that is be mounted as volume for the docker container.
- `/odtp/odtp-output`: Output folder that is mounted as volume for the docker container.
- `/odtp/odtp-logs`: Folder reserved for internal loggings. 
- `/odtp/odtp-config`: Folder reserved for odtp configuration. 

## Testing the component

There are 3 main ways in which you can test a component and the different odtp features. 

1. Testing it as a docker container
2. Testing it as a single component using `odtp`
3. Testing it in a `odtp` digital twin execution

When developing we recomend to start by testing the component via docker and then follow with the others.  

### Testing the component as a docker container

The user will need to manually create the input/output folders and build the docker image.

1. Prepare the following folder structure:

```
- testing-folder
    - data-input
    - data-output
```

Place all required input files in `testing-folder/data-input`.

2. Create your `.env` file with the following parameters.

```
# ODTP COMPONENT VARIABLES
PARAMETER-A=.....
PARAMETER-B=.....
```

3. Build the dockerfile. 

```
docker build -t odtp-component .
```

4. Run the following command.

```
docker run -it --rm \ 
-v {PATH_TO_YOUR_INPUT_VOLUME}:/odtp/odtp-input \
-v {PATH_TO_YOUR_INPUT_VOLUME}:/odtp/odtp-output \
--env-file .env \
odtp-component
```

This command will run the component. If you want debug some errors and execute the docker in an interactive manner, you can use the flag `--entrypoint bash` when running docker.

Also if your tool is interactive such as an Streamlit app, don't forget to map the ports by using `-p XXXX:XXXX`. 

### Testing the component as part of odtp

To execute the command as part of `odtp` please refer to our `odtp` documentation: 

https://odtp-org.github.io/odtp-manuals/

## `odtp.yml`

ODTP requires a set of metadata to work. These fields should be filled by the developers.

```yml
# This file should contain basic component information for your component.
component-name: Component Name
component-author: Component Author
component-version: Component Version
component-repository: Component Repository
component-license: Component License
component-type: ephemeral or interactive
component-description: Description
tags:
  - tag1
  - tag2

# Information about the tools
tools:
  - tool-name: tool's name
    tool-author: Tool's author
    tool-version: Tool version
    tool-repository: Tool's repository
    tool-license: Tool's license

# If your tool require some secrets token to be passed as ENV to the component
# This won't be traced
secrets:
  - name: Key of the argument
  - description: Description of the secret

# If the tool requires some building arguments such as Matlab license
build-args:
  - name: Key of the argument
  - description: Descriptio of the building argument
  - secret: Bool

# If applicable, ports exposed by the component
# Include Name, Description, and Port Value for each port
ports:
  - name: PORT A
    description: Description of Port A
    port-value: XXXX
  - name: PORT B
    description: Description of Port B
    port-value: YYYY

# If applicable, parameters exposed by the component
# Datatype can be str, int, float, or bool.
parameters:
  - name: PARAMETER A
    default-value: DEFAULT_VALUE_A
    datatype: DATATYPE_A
    description: Description of Parameter A
    parameter-bounds: # Boundaries for int and float datatype
      - 0 # Lower bound
      - inf # Upper bound
    options: null
    allow-custom-value: false # If true the user can add a custom value out of parameter-bounds, or options

  - name: PARAMETER B
    default-value: DEFAULT_VALUE_B
    datatype: DATATYPE_B
    description: Description of Parameter B
    parameter-bounds: null
    options: # If your string parameter is limited to a few option, please list them here. 
      - OptionA
      - OptionB
      - OptionC
    allow-custom-value: false # If true the user can add a custom value out of parameter-bounds, or options

# If applicable, data-input list required by the component
data-inputs:
  - name: INPUT A
    type: TYPE_A # Folder or filetype
    path: VALUE_A  
    description: Description of Input A
  - name: INPUT B
    type: TYPE_B # Folder or filetype
    path: VALUE_B  
    description: Description of Input B

# If applicable, data-output list produced by the component
data-output:
  - name: OUTPUT A
    type: TYPE_A # Folder or filetype
    path: VALUE_A
    description: Description of Output A
  - name: OUTPUT B
    type: TYPE_B # Folder or filetype
    path: VALUE_B
    description: Description of Output B

# If applicable, path to schemas to perform semantic validation.
# Still under development. Ignore.
schema-input: PATH_TO_INPUT_SCHEMA
schema-output: PATH_TO_OUTPUT_SCHEMA

# If applicable, define devices needed such as GPU.
devices:
  gpu: Bool
```

## Changelog

- v0.4.0
  - Update default Base and Python in Dockerfile to `ubuntu:22.04` and `python3.10`

- v0.3.4
  - Inclusion of `secrets` and `build-args` in `odtp.yml`
  - Tools as list

- v0.3.3
  - Inclusion of boundaries conditions and options in `odtp.yml` parameters.    

- v0.3.2
  - Extended `odtp.yml` parameters and input/output definition.
  - `odtp.requirements.txt` transfered to submodule `odtp-component-client`.

- v0.3.1
  - Updating schema fields in `odtp.yml` to kebab-case.

- v0.3.0
  - Turning `odtp-client` into a separate repository and adding it as a submodule in `odtp-component-client`
  - Updating `app.sh` and tutorial.
  - Updating `odtp.yml` file.
  - Adding `.DS_Store` to `.gitignore`

- v0.2.0
  - Compatible with ODTP v.0.2.0 only with platform / components
  - Compatible with configuration text files
  - Improved loging system
  - Accepting Digital Twins, Executions, and steps, metadata.
  - Including component versioning in `odtp.yml`

- v0.1.0
  - Compatible with ODTP v.0.1.0 only with platform / components
  - Compatible with configuration text files

## Acknowledgments, Copyright, and Licensing

### Acknowledgments and Funding

This work is part of the broader project **O**pen **D**igital **T**win **P**latform of the **S**wiss **M**obility **S**ystem (ODTP-SMS) funded by Swissuniversities CHORD grant Track B - Establish Projects. ODTP-SMS project is a joint endeavour by the Center for Sustainable Future Mobility - CSFM (ETH Zürich) and the Swiss Data Science Center - SDSC (EPFL and ETH Zürich). 
The Swiss Data Science Center (SDSC) develops domain-agnostic standards and containerized components to manage digital twins. This includes the creation of the Core Platform (both back-end and front-end), Service Component Integration Templates, Component Ontology, and the Component Zoo template. 
The Center for Sustainable Future Mobility (CSFM) develops mobility services and utilizes the components produced by SDSC to deploy a mobility digital twin platform. CSFM focuses on integrating mobility services and collecting available components in the mobility zoo, thereby applying the digital twin concept in the realm of mobility.
 
### Copyright

Copyright © 2023-2024 Swiss Data Science Center (SDSC), www.datascience.ch. All rights reserved.
The SDSC is jointly established and legally represented by the École Polytechnique Fédérale de Lausanne (EPFL) and the Eidgenössische Technische Hochschule Zürich (ETH Zürich). This copyright encompasses all materials, software, documentation, and other content created and developed by the SDSC.

### Intellectual Property (IP) Rights

The Open Digital Twin Platform (ODTP) is the result of a collaborative effort between ETH Zurich (ETHZ) and the École Polytechnique Fédérale de Lausanne (EPFL). Both institutions hold equal intellectual property rights for the ODTP project, reflecting the equitable and shared contributions of EPFL and ETH Zürich in the development and advancement of this initiative.  
 
### Licensing

The Service Component Integration Templates within this repository are licensed under the BSD 3-Clause "New" or "Revised" License. This license allows for broad compatibility and standardization, encouraging open use and contribution. For the full license text, please see the LICENSE file accompanying these templates.

#### Distinct Licensing for Other Components

- **Core Platform**: Open-source under AGPLv3.
- **Ontology**: Creative Commons Attribution-ShareAlike (CC BY-SA).
- **Component Zoo Template**: BSD-3 license.

### Alternative Commercial Licensing

Alternative commercial licensing options for the core platform and other components are available and can be negotiated through the EPFL Technology Transfer Office (https://tto.epfl.ch) or ETH Zürich Technology Transfer Office (https://ethz.ch/en/industry/transfer.html).

## Ethical Use and Legal Compliance Disclaimer

Please note that this software should not be used to deliberately harm any individual or entity. Users and developers must adhere to ethical guidelines and use the software responsibly and legally. This disclaimer serves to remind all parties involved in the use or development of this software to engage in practices that are ethical, lawful, and in accordance with the intended purpose of the software.
