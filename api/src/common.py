import string
import os
import numpy as np

AVAILABLE_CHARS = string.ascii_lowercase + string.digits + ' '
DEST_SHAPE = (240, 320)
VIDEO_EXTENSION = ".mp4"
FRAME_RATE = 23.976

SHOW_DEBUG_PRINT = False
SHOW_DEBUG_IMAGES = False

MODEL_FOLDER = 'models'
HANDS_SEGMENTATION_FOLDER = os.path.join(MODEL_FOLDER, 'hand_segmentation')

ARCHITECTURE_JSON_NAME = 'architecture.json'
WEIGHTS_H5_NAME = 'weights.h5'

MEAN_VALUE_VGG = np.array([103.939, 116.779, 123.68])
