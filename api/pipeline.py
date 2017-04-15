from PIL import Image
from binascii import a2b_base64
from src.utils.Transformers.basic_morpho_transforms import Resize
from src.utils.Transformers.eval_transformer_pipeline import eval_transformer_pipeline as eval
import io
import cv2
import re
import numpy as np


def get_transformers():
    transformers = [
        Resize(width=1024, height=768)
    ]

    return transformers


def get_letter(video_sequence):
    transformers = get_transformers()
    letter = eval(video_sequence, transformers)
    return letter


def load_img(img_str):
    image_data = re.sub('^data:image/.+;base64,', '', img_str)
    buffer = io.BytesIO()
    buffer.write(a2b_base64(image_data))
    img = Image.open(buffer)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return img


def process_json_img_array(json_array):
    sequence = []
    for img_string in json_array:
        sequence.append(load_img(img_string))
    return sequence


def evaluate(json_array):
    sequence = process_json_img_array(json_array)
    return get_letter(sequence)