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
import sys
import time

def save_audio(json_response, id):
    parts = id.split('/')
    last = parts[-1]
    result = last.split('.')[0]
    print("=====================================================")
    print(f"Technically we should save {result} but we do not have the storage to save all the audio files so let's just sleep for a random amount of time here, then continue")
    sleep = random.randint(1,9)
    time.sleep(sleep/10)
    print("=====================================================")
    # torchaudio.save(f"output_files/{result}.wav", torch.tensor(json_response).unsqueeze(0), 16000)


if __name__ == "__main__":
    # Read JSON data from standard input (stdin)

    parser = argparse.ArgumentParser(description="Process JSON data with custom arguments")

    parser.add_argument("id", help="Response ID")
    args = parser.parse_args()

    try:
        data = json.load(sys.stdin)
        save_audio(data, args.id)
    except json.JSONDecodeError:
        print("Error: Invalid JSON data received.")