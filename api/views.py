from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import json
import av
import numpy as np
import torch
from transformers import VivitImageProcessor, VivitForVideoClassification, VivitConfig
import requests
import io
import os

np.random.seed(0)


def read_video_pyav(container, indices):
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices


def vivit(file_path):
    response = requests.get(file_path, stream=True)
    file_stream = io.BytesIO(response.content)
    container = av.open(file_stream)
    indices = sample_frame_indices(clip_len=32, frame_sample_rate=4, seg_len=container.streams.video[0].frames)
    video = read_video_pyav(container=container, indices=indices)
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'api/config.json')
    weights_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'api/pytorch_model.bin')
    preprocessor_config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'api/preprocessor_config.json')
    model_config = VivitConfig.from_pretrained(config_path)
    model = VivitForVideoClassification(model_config)
    model.load_state_dict(torch.load(weights_path))
    image_processor = VivitImageProcessor.from_pretrained(preprocessor_config_path)
    inputs = image_processor(list(video), return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    predicted_label = logits.argmax(-1).item()
    return model.config.id2label[predicted_label]


class VivitAPI(APIView):
    def post(self, request):
        try:
            request_data = json.loads(request.body.decode('utf-8'))
            file_path = request_data.get('file_path')
            result = vivit(file_path)
            response_data = {'result': result}
            return Response(response_data, status=status.HTTP_200_OK)
        except Exception as e:
            error_message = str(e)
            return Response({'error': error_message}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
