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
HANDS_SEGMENTATION_HAAR_FOLDER = os.path.join(HANDS_SEGMENTATION_FOLDER, 'haarcascades')
GEST_HAAR_CASCADE = os.path.join(HANDS_SEGMENTATION_HAAR_FOLDER, 'aGest.xml')
CLOSED_PALM_CASCADE = os.path.join(HANDS_SEGMENTATION_HAAR_FOLDER, 'closed_frontal_palm.xml')
OVERALL_PALM_CASCADE = os.path.join(HANDS_SEGMENTATION_HAAR_FOLDER, 'palm2.xml')
CUSTOM_PALM_CASCADE = os.path.join(HANDS_SEGMENTATION_HAAR_FOLDER, 'Hand.Cascade.1.xml')

GESTURE_PREDICTION_FOLDER = os.path.join(MODEL_FOLDER, 'gesture_classification')
PREDICTION_SELECTION_FOLDER = os.path.join(MODEL_FOLDER, 'prediction_selection')
CHAR_PREDICTION_FOLDER = os.path.join(MODEL_FOLDER, 'char_prediction')

ARCHITECTURE_JSON_NAME = 'architecture.json'
WEIGHTS_H5_NAME = 'weights.h5'
WEIGHTS_HDF5_NAME = 'weights.hdf5'

MEAN_VALUE_VGG = np.array([103.939, 116.779, 123.68])
CLASSIFIER_INPUT_SHAPE = (64, 64)
