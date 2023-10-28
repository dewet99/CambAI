import requests
import argparse
import yaml
import json
import os
import random
import numpy as np
import traceback
import torchaudio
import torch

def send_audio_to_server(server_url, json_dict, id):
    headers = {'Content-Type': 'application/json'}
    response = requests.post(server_url, data=json_dict)

    if response.status_code == 200:
        # result = response.json()
        # Process the inference result
        print("======================================")
        print(f"Inference successful for request number {id}")
        print("======================================")
        json_response = response.json()
        torchaudio.save(f"output_files/output_{id}.wav", torch.tensor(json_response).unsqueeze(0), 16000)

    else:
        # Handle errors or log them
        print(f"Request failed with status code: {response.status_code}")

def generate_json_files_for_inference(dataset_path):
    
    num_readers = len(next(os.walk(dataset_path))[1]) # number of folders, each is a unique reader

    # select a random reader
    directories = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]

    src_reader_id = np.random.randint(0,num_readers) #randomly select a reader id for the source
    src_reader_dir = directories[src_reader_id] # reader dir for the source utterances

    target_reader_id = np.random.randint(0,num_readers)

    while target_reader_id == src_reader_id:
        target_reader_id = np.random.randint(0,num_readers) #so src and target aren't the same

    target_reader_dir = directories[target_reader_id] # reader dir for the target utterances

    # src_num_chapters_avail = len(next(os.walk(f"{dataset_path}/{src_reader_dir}"))[1]) #how many folders the src reader has to select from
    # target_num_chapters_avail = len(next(os.walk(f"{dataset_path}/{target_reader_dir}"))[1]) #how many folders the target reader has to select from

    src_chapter_dirs = [d for d in os.listdir(f"{dataset_path}/{src_reader_dir}") if os.path.isdir(os.path.join(f"{dataset_path}/{src_reader_dir}", d))] #the actual names of the chapter dirs for the chosen speaker
    target_chapter_dirs = [d for d in os.listdir(f"{dataset_path}/{target_reader_dir}") if os.path.isdir(os.path.join(f"{dataset_path}/{target_reader_dir}", d))] #the actual names of the chapter dirs for the chosen target

    src_chapter = random.choice(src_chapter_dirs)
    target_chapter = random.choice(target_chapter_dirs)

    src_path = f"{dataset_path}/{src_reader_dir}/{src_chapter}"
    target_path = f"{dataset_path}/{target_reader_dir}/{target_chapter}"

    # choosing a random .flac file for the src utterance
    src_files = [f for f in os.listdir(src_path) if os.path.isfile(os.path.join(src_path, f))]
    src_flac_files =[f for f in src_files if f.endswith('.flac')]
    src_file = random.choice(src_flac_files)

    # choosing a random number of .flac files for the target utterance
    target_files = [f for f in os.listdir(target_path) if os.path.isfile(os.path.join(target_path, f))]
    target_flac_files =[f for f in target_files if f.endswith('.flac')]

    print(f"There are {len(target_flac_files)} files in the directory to select from")
    num_files_to_select = np.random.randint(1, len(target_flac_files)-1)

    target_files = random.sample(target_flac_files,num_files_to_select)

    src_path_final = f"{dataset_path}/{src_reader_dir}/{src_chapter}/{src_file}"
    target_paths_final = [f"{dataset_path}/{target_reader_dir}/{target_chapter}/{target_file}" for target_file in target_files]

    data = {
        "source_path": src_path_final,
        "target_paths": target_paths_final,
    }

    return data


def convert_json_paths_to_json_lists(path):

    audio_dict = {}

    source_audio, _ = torchaudio.load(path["source_path"], normalize=True)
    audio_dict["source_audio"] = source_audio.tolist()[0]

    target_audios = []
    for id, path in enumerate(path["target_paths"]):
        ta, _ = torchaudio.load(path, normalize=True)
        target_audios.append(ta.tolist()[0])

    audio_dict["target_audios"] = target_audios

    # json_file_path = "temp_dict.json"

    # with open(json_file_path, "w") as json_file:
    #     json.dump(audio_dict, json_file, ensure_ascii=False)

    return audio_dict





def main():

    parser = argparse.ArgumentParser(description="Send audio files to TorchServe for inference.")
    parser.add_argument("model_name", help = "Specify the model name as on torchserve")
    parser.add_argument("dataset_relative_path", help = "Path to the dataset test-clean relative to the current working directory")
    parser.add_argument("id", help = "Request id")
    args = parser.parse_args()

    # dataset_path = "/media/Data/CambAI/TakeHome/CambAI_TakeHome/datasets/LibriSpeech/test-clean"

    dataset_relative_path = args.dataset_relative_path
    dataset_path = f"{os.getcwd()}/{dataset_relative_path}" 
    print(dataset_path)
    # TorchServe server URL and model endpoint
    # server_url = f"http://localhost:8080/predictions/{args.model_name}"
    # server_url = f"http://localhost:8081/predictions/model"



    # Load audio file paths from the provided YAML file
    try:
            paths  = generate_json_files_for_inference(dataset_path)
            data = convert_json_paths_to_json_lists(paths)
            
            # sending to server via async curl thingies
            with open("input_data.json", "w") as json_file:
                json.dump(data, json_file)
            
            # Sending to server via python script:
            # let's manually convert to json
            # json_string = json.dumps(data)
            # byte_array = json_string.encode('utf-8')
            # send_audio_to_server(server_url, byte_array, args.id)
        



    except Exception as e:
        traceback.print_exc()
        return
        


    

    # Adjust to your server and model endpoint


    

if __name__ == "__main__":
    main()