from PIL import Image
from binascii import a2b_base64
from src.utils.Transformers.CNN_utils.transforms import HandsLocalizer, GestureClassifier, CharPredictor, PredictionSelector
from src.utils.Transformers.basic_morpho_transforms import Resizer
from src.utils.Transformers.eval_transformer_pipeline import eval_transformer_pipeline as eval
from src.utils.Transformers.eval_transformer_pipeline import eval_transformer_pipeline_store_all as eval_with_every_stage
import io
import cv2
import re
import numpy as np


def get_transformers():
    transformers = [
        Resizer(width=320, height=240),
        HandsLocalizer(),
        GestureClassifier(),
        CharPredictor(num_of_chars=1),
        PredictionSelector(indices_of_transformers_to_combine=[2, 3])
    ]

    return transformers


def get_letter(video_sequence):
    transformers = get_transformers()
    letter = eval(video_sequence, transformers)
    return letter


def get_stages(video_sequence):
    transformers = get_transformers()
    stages = eval_with_every_stage(video_sequence, transformers)
    return stages


def load_img(img_str):
    image_data = re.sub('^data:image/.+;base64,', '', img_str)
    buffer = io.BytesIO()
    buffer.write(a2b_base64(image_data))
    img = Image.open(buffer)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    del buffer
    return img


def process_json_img_array(json_array):
    sequence = []
    for img_string in json_array:
        sequence.append(load_img(img_string))
    return sequence


def evaluate(json_array, stages=True):
    sequence = process_json_img_array(json_array)
    return get_letter(sequence) if not stages else get_stages(sequence)
