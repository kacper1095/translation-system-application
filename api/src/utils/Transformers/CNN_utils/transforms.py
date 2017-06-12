from __future__ import print_function

from ..Transformer import Transformer
from ...AsciiEncoder import AsciiEncoder
from .mocks import GestureClassifierMock, CharPredictionMock, PredictionSelectionMock, HandsLocalizerMock

from keras.utils.np_utils import to_categorical
from keras.models import model_from_json, load_model
from src.common import (
    HANDS_SEGMENTATION_FOLDER, ARCHITECTURE_JSON_NAME, WEIGHTS_H5_NAME, CHAR_PREDICTION_FOLDER, WEIGHTS_HDF5_NAME,
    CLOSED_PALM_CASCADE, GEST_HAAR_CASCADE, OVERALL_PALM_CASCADE, CLASSIFIER_INPUT_SHAPE, GESTURE_PREDICTION_FOLDER,
    CUSTOM_PALM_CASCADE
)
from src.utils.Logger import Logger
from .helpers import Coordinates

import sys
import numpy as np
import os
import cv2


class CNNTransformer(Transformer):
    transformers = []

    CACHE_MAX_SIZE = 5
    MAX_FRAME_COUNTER = 3

    def __init__(self):
        super(CNNTransformer, self).__init__()
        self.model = None
        self.cache = []
        self.frame_counter = 0

    def clear_cache(self):
        self.cache.clear()
        self.frame_counter = 0

    def load_model(self):
        pass


class HandsLocalizerTracker(CNNTransformer):
    def __init__(self):
        super(HandsLocalizerTracker, self).__init__()
        self.closed_hand_cascade = cv2.CascadeClassifier(CLOSED_PALM_CASCADE)
        self.open_hand_cascade = cv2.CascadeClassifier(OVERALL_PALM_CASCADE)
        self.gest_cascade = cv2.CascadeClassifier(GEST_HAAR_CASCADE)
        self.third_hand_cascade = cv2.CascadeClassifier(os.path.join(CUSTOM_PALM_CASCADE))
        self.out_key = 'hands'

        self.tracker = None
        self.coordinates = Coordinates()
        self.tracker_init = False
        self.bbox = None
        self.counter = 0
        self.min_size = (50, 50)  # forged opencv parameter
        self.scale_factor = 1.1     # forged opencv parameter
        self.min_neighbors = 3      # forged opencv parameter
        self.tracker = None
        self.fitting_with_cascades_frequency = 40

    @staticmethod
    def threshold_image(img):
        img = img.astype('uint8')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        ret, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        row, col = np.where(thresholded != 0)
        left = col.min()
        right = col.max()
        mean = (left + right)//2
        # img[thresholded == 255] = 0
        img = img[:, mean - mean//2: mean + mean//2]
        return img

    def transform(self, X, **transform_params):
        self.coordinates.max_height = X.shape[1]
        self.coordinates.max_width = X.shape[2]
        offset = 0
        for frame in X.astype('uint8'):
            if not self.tracker_init or self.counter == self.fitting_with_cascades_frequency:
                self.counter = 0
                self.coordinates.clear_all()
                closed_hands = self.closed_hand_cascade.detectMultiScale(frame, self.scale_factor, self.min_neighbors,
                                                                         cv2.CASCADE_FIND_BIGGEST_OBJECT,
                                                                         minSize=self.min_size)
                open_hands = self.open_hand_cascade.detectMultiScale(frame, self.scale_factor, self.min_neighbors,
                                                                     cv2.CASCADE_FIND_BIGGEST_OBJECT,
                                                                     minSize=self.min_size)
                gest = self.gest_cascade.detectMultiScale(frame, self.scale_factor, self.min_neighbors,
                                                          cv2.CASCADE_FIND_BIGGEST_OBJECT,
                                                          minSize=self.min_size)
                custom_hands = self.third_hand_cascade.detectMultiScale(frame, self.scale_factor, self.min_neighbors,
                                                                        cv2.CASCADE_FIND_BIGGEST_OBJECT,
                                                                        minSize=self.min_size)
                hands = list(open_hands) + list(closed_hands) + list(gest) + list(custom_hands)

                if len(hands) == 0 and self.coordinates.has_cords():
                    x, y, w, h = self.coordinates.get_processed_cords()
                else:
                    for hand in hands:
                        self.coordinates.add_hand(hand)
                    x, y, w, h = self.coordinates.get_processed_cords()

                partial_frame = frame[y:y+h, x:x+w]
                if not HandsLocalizerTracker.is_shape_zero(partial_frame.shape):
                    bbox = (x, y, w, h)
                    self.tracker = cv2.Tracker_create("MIL")
                    self.tracker_init = self.tracker.init(frame, bbox)
            else:
                self.tracker_init, bbox = self.tracker.update(frame)
                x, y, w, h = bbox
                x += offset
                w -= offset
                h -= offset
                self.coordinates.add_hand((x, y, w, h))
                self.counter += 1

        x, y, w, h = self.coordinates.get_processed_cords()
        output = X.copy()
        transformed = []
        for out in output:
            sliced = out[y:y+h, x:x+w]
            if HandsLocalizerTracker.is_shape_zero(sliced.shape):
                sliced = out
            thresholded = HandsLocalizerTracker.threshold_image(sliced)
            scaled = cv2.resize(thresholded, CLASSIFIER_INPUT_SHAPE, interpolation=cv2.INTER_CUBIC)
            transformed.append(scaled)
        self.output = np.asarray(transformed)
        return self.output

    @staticmethod
    def is_shape_zero(shape):
        for axis in shape:
            if axis == 0:
                return True
        return False


class HandsLocalizer(CNNTransformer):
    def __init__(self):
        super(HandsLocalizer, self).__init__()
        self.model = self.load_model()
        self.out_key = 'hands'

    def transform(self, X, **transform_params):
        last_frame = np.asarray([X.transpose(0, 3, 1, 2)[-1]])
        prediction = self.model.predict(last_frame)
        prediction[prediction > 0.1] = 1.
        Logger.log("output_shape: " + str(prediction.shape))
        # output = X.transpose(0, 3, 1, 2)
        output = X.copy() * 255.
        prediction_colored = cv2.cvtColor((prediction[0][0] * 255), cv2.COLOR_GRAY2BGR)
        cv2.imwrite('text.jpg', prediction_colored)
        Logger.log("color shape: " + str(prediction_colored.max()))
        output[-1] = prediction_colored
        self.output = output
        return self.output

    def load_model(self):
        with open(os.path.join(HANDS_SEGMENTATION_FOLDER, ARCHITECTURE_JSON_NAME), 'r') as f:
            model = model_from_json(f.read())
        model.load_weights(os.path.join(HANDS_SEGMENTATION_FOLDER, WEIGHTS_H5_NAME))
        return model
        # return HandsLocalizerMock.model()


class GestureClassifier(CNNTransformer):
    def __init__(self):
        super(GestureClassifier, self).__init__()
        self.model = self.load_model()
        self.out_key = 'gesture'

    def transform(self, X, **transform_params):
        if len(self.cache) < CNNTransformer.CACHE_MAX_SIZE:
            if self.frame_counter == CNNTransformer.MAX_FRAME_COUNTER:
                self.cache.append(X[-1])
                self.frame_counter = 0
            else:
                self.frame_counter += 1
            return None

        inp = np.array(self.cache).transpose((0, 3, 1, 2))
        # inp = np.clip(inp * 2.5, 0, 1)
        prediction = self.model.predict(inp)
        # prediction = self.model.predict(inp[np.newaxis].transpose((0, 2, 1, 3, 4)))
        self.output = prediction[-1]
        self.clear_cache()
        return self.output

    def load_model(self):
        with open(os.path.join(GESTURE_PREDICTION_FOLDER, ARCHITECTURE_JSON_NAME)) as f:
            model = model_from_json(f.read())
        model.load_weights(os.path.join(GESTURE_PREDICTION_FOLDER, WEIGHTS_H5_NAME))
        return model
        # return load_model(os.path.join(GESTURE_PREDICTION_FOLDER, '90per.h5'))
        # return GestureClassifierMock.model()


class CharPredictor(CNNTransformer):
    __previous_predictions = to_categorical(np.random.randint(len(AsciiEncoder.AVAILABLE_CHARS), size=20),
                                            len(AsciiEncoder.AVAILABLE_CHARS))

    def __init__(self, num_of_chars):
        super(CharPredictor, self).__init__()
        self.num_of_chars = num_of_chars
        self.model = self.load_model()
        self.out_key = 'chars'

    def transform(self, X, **transform_params):
        if len(self.cache) < CNNTransformer.CACHE_MAX_SIZE:
            if self.frame_counter == CNNTransformer.MAX_FRAME_COUNTER:
                self.cache.append(True)
                self.frame_counter = 0
            else:
                self.frame_counter += 1
            return None
        x_data = CharPredictor.__previous_predictions
        self.output = self.model.predict(np.asarray([x_data]))[0]
        self.clear_cache()
        return self.output

    def prepare_data(self):
        predictions = CharPredictor.__previous_predictions[:]
        if len(predictions) >= self.num_of_chars:
            while len(predictions) > self.num_of_chars:
                predictions.pop(0)
        else:
            while len(predictions) != self.num_of_chars:
                predictions.insert(0, 0)
        return to_categorical(predictions, len(AsciiEncoder.AVAILABLE_CHARS))

    @staticmethod
    def add_to_previous_predictions(prediction):
        CharPredictor.__previous_predictions = np.append(CharPredictor.__previous_predictions[1:],
                                                         to_categorical(prediction, len(AsciiEncoder.AVAILABLE_CHARS)),
                                                         axis=0)

    def load_model(self):
        with open(os.path.join(CHAR_PREDICTION_FOLDER, ARCHITECTURE_JSON_NAME), 'r') as f:
            model = model_from_json(f.read())
        model.load_weights(os.path.join(CHAR_PREDICTION_FOLDER, WEIGHTS_HDF5_NAME))
        return model
        # return CharPredictionMock.model(time_steps=self.num_of_chars, feature_length=len(AsciiEncoder.AVAILABLE_CHARS))


class PredictionSelector(CNNTransformer):
    def __init__(self, indices_of_transformers_to_combine):
        super(PredictionSelector, self).__init__()
        self.model = self.load_model()
        self.indices_of_transformers_to_combine = indices_of_transformers_to_combine
        self.out_key = 'prediction_selection'

    def transform(self, X, **transform_params):
        if len(self.cache) < CNNTransformer.CACHE_MAX_SIZE:
            if self.frame_counter == CNNTransformer.MAX_FRAME_COUNTER:
                self.cache.append(True)
                self.frame_counter = 0
            else:
                self.frame_counter += 1
            return None
        gesture_prediction = CNNTransformer.transformers[self.indices_of_transformers_to_combine[0]].output
        char_prediction = CNNTransformer.transformers[self.indices_of_transformers_to_combine[1]].output
        # x_data = np.asarray([[previous_transformer_output]])
        # prediction = self.model.predict(x_data)
        # prediction = x_data[0][0][0]
        prediction = gesture_prediction * 0.8 + char_prediction * 0.2
        # prediction = gesture_prediction
        predicted_arg = np.argmax(prediction)
        CharPredictor.add_to_previous_predictions(predicted_arg)
        prediction = np.array([prediction, char_prediction])
        self.output = prediction
        self.clear_cache()
        return self.output

    def fit_transform(self, X, y=None, **fit_params):
        previous_transformer_output = []
        for index in self.indices_of_transformers_to_combine:
            previous_transformer_output.append(CNNTransformer.transformers[index].output)
        x_data = np.asarray([[previous_transformer_output]])
        prediction = self.model.fit(x_data, y)
        CharPredictor.add_to_previous_predictions(prediction[0][0])

    def load_model(self):
        return PredictionSelectionMock.model()
