# Name of the component

Description of the component

| Tool Info | Links |
| --- | --- |
| Original Tool | []() |
| Current Tool Version  | [commit-hash](link-to-commit-hash) |


## ODTP command 

```
odtp new odtp-component-entry \
--name odtp-component \
--component-version x.y.z \
--repository Link to repository
``` 

## Data sheet

### Parameters

| Parameter | Description | Default Value |
| --- | --- | --- |
| A | B | C |

### Input Files

| File/Folder | Description |
| --- | --- | 
| A | B |

### Output Files

| File/Folder | Description |
| --- | --- | 
| A | B |

## Tutorial

### How to run this component as docker

Build the dockerfile 

```
docker build -t odtp-component .
```

Run the following command. Mount the correct volumes for input/output folders. 

```
docker run -it --rm \
-v {PATH_TO_YOUR_INPUT_VOLUME}:/odtp/odtp-input \
-v {PATH_TO_YOUR_OUTPUT_VOLUME}:/odtp/odtp-output \
--env-file .env odtp-component
```


## Developed by

XXXXXXXXXX
