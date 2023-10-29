from ts.torch_handler.base_handler import BaseHandler
import torch
import json
import os
import torch, torchaudio
# import IPython.display as ipd
import urllib.parse
import base64
import torch.profiler as prof

class KNN_VC_Handler(BaseHandler):
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
        os.environ['LRU_CACHE_CAPACITY'] = '1' # This supposedly helps with memory leaks when inputs are of varying lengths 

        # From the exapmle:
        self.manifest = context.manifest
        properties = context.system_properties
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")

        # Download model
        download_dir = "/home/model-server/knn-vc-downloads"
        torch.hub.set_dir(download_dir)
        self.model = torch.hub.load('bshall/knn-vc', 'knn_vc', prematched=True, trust_repo=True, pretrained=True, device=self.device)

        self.initialized = True

    def preprocess(self, data):

        inputs = data[0].get("body") # data is bytearray from json file 
        if inputs == None:
            inputs = data[0].get("data")

        inputs = inputs.decode('utf-8')
        data = json.loads(inputs)


        # source_audio_paths = data["source_path"]
        source_audio = data["source_audio"]
        target_audio = data["target_audios"]

        # convert the target_audio list of lists into a list of tensors:
        targets = [torch.tensor(target) for target in target_audio]

        query_seq = self.model.get_features(torch.tensor(source_audio))
        matching_set = self.model.get_matching_set(targets)

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
        out_wav = self.postprocess(out_wav)
        return [out_wav]


    def postprocess(self, out_wav):
        out_wav_list = out_wav.tolist()
        return out_wav_list


    def inference(self, query, match):
        infer = self.model.match(query, match, topk=4)
        return infer
        
        



        