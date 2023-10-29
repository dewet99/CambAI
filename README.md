# CambAI Takehome Challenge

A brief description of your project.

## Table of Contents

- [Project Name](#project-name)
- [Description](#description)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Description

Provide a more detailed description of your project here. Explain its purpose, features, and any other relevant information.

## Getting Started
I created a docker container with everything ready to go. Run the following commands:

```shell
# get the container image
docker pull dewet99/knn_vc_camb_wm:latest

# run it 
sudo docker run --gpus 1 --rm -it -p 127.0.0.1:8080:8080 -p 127.0.0.1:8081:8081 -p 127.0.0.1:8082:8082 --name knn_vc dewet99/knn_vc_camb_wm:latest

# register the knn_vc models - it is already included in the container
curl -X POST "localhost:8081/models?model_name=knn_vc&url=/home/model-server/model-store/model_store/knn_vc.mar&initial_workers=1"

```
Succesfully registering the models should show the following in the terminal:
```shell
{
  "status": "Model \"knn_vc\" Version: 1.0 registered with 1 initial workers"
}
```
You can now start doing inference.

## Inference
We have two kinds of inputs: Noise and audio. Running a stress test with audio will cause an out of memory error, because the number of inputs can become quite high. We used random noise as source and targets for inference.
```shell
bash ./simulate_stress_test.sh --num_requests --model_name --dataset_relative_path --input_type

```
The args are as follows:
1. `--num_requests`: How many inference requests to generate
2. `--model_name`: knn_vc, it's the only one in the container
3. `--dataset_relative_path`: Path to the LibriSpeech dataset for inferring with actual audio
4. `--input_type`: noise or audio
   
The following command will run a stress test wtih 1000 inference requests, consisting of source and target clips of random noise:
```shell
bash ./simulate_stress_test.sh 1000 knn_vc ./datasets/LibriSpeech/test-clean noise
```

Inferring with actual audio requires the LibriSpeech test-clean set. You can download and install it with the following command:
```shell
bash ./get_speech
```
In theory, that should download and extract the dataset to the correct location. Can then do a stress test with real audio using the following command:
```shell
bash ./simulate_stress_test.sh 1000 knn_vc ./datasets/LibriSpeech/test-clean audio
```
This caused my PC to run out of memory, I did not have time to investigate whether it was a memory leak or just due to incorrect config options in terms of allocated RAM.
The output data does not get saved, for either noise or real audio. Simply comment and uncomment the relevant lines in the `simulate_stress_test.sh` script to change this. We had some storage issues, hence this measure.

### Prerequisites

List the software and tools that users need to have installed before they can use your project. Include version numbers if necessary.
