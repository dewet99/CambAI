from ts.torch_handler.base_handler import BaseHandler
import torch
import json
import os
import torch, torchaudio
import IPython.display as ipd
import urllib.parse
try:
    from dummy_model import DummyModel
except:
    print('========================================')
    print("Couldn't import dummy model, but is okey")
    print('========================================')
# from hubconf import knn_vc


class DummyHandler(BaseHandler):
    """
    Refer to https://pytorch.org/serve/custom_service.html
    """
    def __init__(self):
        self._context = None
        self.initialized = None
        self.model = None
        self.device = None


    def initialize(self,context):
        """
        Initialize model and resources here
        """
        # From the exapmle:
        self.manifest = context.manifest
        properties = context.system_properties
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")

        # Download model
        # download_dir = "~/media/Data/CambAI/TakeHome/CambAI_Takehome/knn-vc-downloads"
        # torch.hub.set_dir(download_dir)
        # self.model = torch.hub.load('bshall/knn-vc', 'knn_vc', prematched=True, trust_repo=True, pretrained=True, device=self.device)

        # Dummy model:
        self.model = DummyModel()

        self.initialized = True

    def preprocess(self, data):
        
        # if using curl to pass json file:
        # data = data[0].get("body") # data is bytearray from json file  
        # data = json.loads(data.decode('utf-8')) # converted to dict

        # if using python script to pass json file
        data = data[0].get("body") # data is bytearray from json file  
        data_str = data.decode('utf-8')
        parsed_data = urllib.parse.parse_qs(data_str)

        decoded_dict = {}
        for key, values in parsed_data.items():
            if len(values) == 1:
                # If a parameter appears only once, assign its value as a string
                decoded_dict[key] = urllib.parse.unquote(values[0])
            else:
                # If a parameter appears multiple times, assign its values as a list
                decoded_dict[key] = [urllib.parse.unquote(val) for val in values]

        # decoded_dict = {key: [urllib.parse.unquote(val) for val in values] for key, values in parsed_data.items()}
        # source_audio_paths = data["source_path"]
        source_audio_paths = decoded_dict["source_path"]

        # target_audios = [data.get(f'target_{i}') for i in range(len(target_audio_paths))]

        # target_audio_paths = data["target_paths"]
        target_audio_paths = decoded_dict["target_paths"]

        print('====================')
        print(target_audio_paths)
        print('====================')

        # ref_wav_paths = data["target_paths"]

        query_seq = self.model.get_features(source_audio_paths) # Returns features of `path` waveform as a tensor of shape (seq_len, dim) --> data preprocessing
        matching_set = self.model.get_matching_set(target_audio_paths) # Get matching features to be used by wavlm --> Data preprocessing

        # print("==========================")
        # print(f"Source audio: {source_audio_paths}")
        # print(f"Target audio: {target_audio_paths}")
        # print("==========================")
        # raise Exception

        return query_seq, matching_set, data

    def handle(self, data, context):
        """
        Invoke by TorchServe for prediction request.
        Do pre-processing of data, prediction using model and postprocessing of prediciton output
            data:   Input data for prediction
            context:    Initial context contains model server system properties.
            return: prediction output
        """

        # First pre-process the data. Accept data as a dict, and get the stuff
        query_seq, matching_set, data = self.preprocess(data)
        out_wav = self.inference(query_seq,matching_set)
        self.postprocess(data, out_wav)
        # return out_wav


    def postprocess(self, data, out_wav):
        ipd.Audio(out_wav.numpy(), rate=16000)
        dir = data["working_dir"]
        parts = data["source_path"].split("/")
        original_filename = parts[-1]
        filename_without_extension = original_filename.split('.')[0]

        path = f"{dir}/output_files/{filename_without_extension}_converted.wav"

        torchaudio.save(path, out_wav[None], 16000)

        print("==========================")
        print(f"saved filename as {path}")
        print("==========================")

    def inference(self, query, match):
        return self.model.match(query, match, topk=4)
        
        



        