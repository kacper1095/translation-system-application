from PIL import Image
from binascii import a2b_base64
from src.utils.Transformers.CNN_utils.transforms import (
    HandsLocalizer, GestureClassifier, CharPredictor, PredictionSelector, CNNTransformer
)
from src.utils.Transformers.basic_morpho_transforms import (
    Resizer, BoxHands, Normalizer, BGR2HSV
)
from src.utils.Transformers.eval_transformer_pipeline import eval_transformer_pipeline as eval
from src.utils.Transformers.eval_transformer_pipeline import eval_transformer_pipeline_store_all as eval_with_every_stage
from src.utils.AsciiEncoder import AsciiEncoder
import io
import cv2
import re
import numpy as np

transformers = None


def load_transformers():
    global transformers
    transformers = [
        Resizer(width=320),
        Normalizer(),
        # BGR2HSV(),
        HandsLocalizer(),
        BoxHands(),
        GestureClassifier(),
        CharPredictor(num_of_chars=20),
        PredictionSelector(indices_of_transformers_to_combine=[4, 5])
    ]
    CNNTransformer.transformers = transformers


def get_letter(video_sequence):
    letter = eval(video_sequence, transformers)
    return letter


def get_stages(video_sequence):
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
    return np.asarray(sequence).astype('float32')


def evaluate(json_array, stages=True):
    sequence = process_json_img_array(json_array)
    return get_letter(sequence) if not stages else get_stages(sequence)


def convert_last_output_to_ascii(last_output):
    index = np.argmax(last_output)
    return AsciiEncoder.convert_indexes_to_characters(index)
