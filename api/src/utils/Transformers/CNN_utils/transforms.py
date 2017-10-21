from __future__ import print_function

from ..Transformer import Transformer
from ...AsciiEncoder import AsciiEncoder
from .mocks import PredictionSelectionMock
from .metrics import f1

from keras.utils.np_utils import to_categorical
from keras.models import model_from_json, load_model
from src.common import (
    HANDS_SEGMENTATION_FOLDER, ARCHITECTURE_JSON_NAME, WEIGHTS_H5_NAME, CHAR_PREDICTION_FOLDER, WEIGHTS_HDF5_NAME,
    CLOSED_PALM_CASCADE, GEST_HAAR_CASCADE, OVERALL_PALM_CASCADE, CLASSIFIER_INPUT_SHAPE, GESTURE_PREDICTION_FOLDER,
    CUSTOM_PALM_CASCADE, PATH_TO_CKPT_BOXING, BOXING_INPUT_TENSOR_NAME, BOXING_OUTPUT_TENSOR_NAME, MIN_THRESHOLD_BOXES
)
from src.utils.Logger import Logger

from .utils import TensorflowWrapper
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


class TensorflowHandsLocalizer(CNNTransformer):
    def __init__(self):
        super(TensorflowHandsLocalizer, self).__init__()
        self.model = self.load_model()
        self.out_key = 'hands'
        self.previous_coordinates = []
        self.max_preious_elements = 10

    def transform(self, X, **transform_params):
        input_image = X[-1]
        inp = np.array([input_image]).transpose((0, 3, 1, 2))
        prediction_boxes, prediction_scores = self.model.predict(inp)
        boxes_with_scores = self.process_output(input_image, prediction_boxes[0], prediction_scores[0])
        if len(boxes_with_scores) == 0 and len(self.previous_coordinates) == 0:
            self.output = np.array(X)
            return self.output
        else:
            hand_boxes = self.cut_hands(input_image, boxes_with_scores, num_of_boxes=1)
            self.output = np.array([hand_boxes[0]])
            return self.output

    def load_model(self):
        return TensorflowWrapper(PATH_TO_CKPT_BOXING, BOXING_INPUT_TENSOR_NAME, BOXING_OUTPUT_TENSOR_NAME)
        # return HandsLocalizerMock.model()

    def process_output(self, image, boxes, scores):
        h, w, c = image.shape
        result_boxes = []
        offset = 20
        for i in range(len(boxes)):
            if scores[i] > MIN_THRESHOLD_BOXES:
                ymin, xmin, ymax, xmax = boxes[i]
                left, right, top, bottom = (xmin * w, xmax * w, ymin * h, ymax * h)
                # left = max(0, left - offset)
                # right = min(w - 1, right + offset)
                # top = min(0, top - offset)
                # bottom = max(h - 1, bottom + offset)
                result_boxes.append((left, right, top, bottom, scores[i]))
        return np.array(result_boxes)

    def cut_hands(self, image, boxes_with_scores, num_of_boxes=1):
        hands = []
        if len(boxes_with_scores) > 0:
            sorted_boxes_with_scores_indices = np.argsort(boxes_with_scores[:, -1])[:num_of_boxes]
            for index in sorted_boxes_with_scores_indices:
                left, right, top, bottom, score = boxes_with_scores[index]
                self.add_elem_to_previous_coordinates((left, right, top, bottom))
        mean_left, mean_right, mean_top, mean_bottom = self.get_mean_coordinates()
        hand = image[int(mean_top):int(mean_bottom), int(mean_left):int(mean_right)]
        hands.append(hand)
        return np.array(hands)

    def add_elem_to_previous_coordinates(self, elem):
        if len(self.previous_coordinates) == self.max_preious_elements:
            self.previous_coordinates.pop(0)
        self.previous_coordinates.append(elem)

    def get_mean_coordinates(self):
        return np.mean(np.array(self.previous_coordinates), axis=0)


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
        self.scale_factor = 1.1  # forged opencv parameter
        self.min_neighbors = 3  # forged opencv parameter
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
        mean = (left + right) // 2
        # img[thresholded == 255] = 0
        img = img[:, mean - mean // 2: mean + mean // 2]
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

                partial_frame = frame[y:y + h, x:x + w]
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
            sliced = out[y:y + h, x:x + w]
            if HandsLocalizerTracker.is_shape_zero(sliced.shape):
                sliced = out
            thresholded = HandsLocalizerTracker.threshold_image(sliced)
            scaled = cv2.resize(thresholded, CLASSIFIER_INPUT_SHAPE, interpolation=cv2.INTER_CUBIC)
            transformed.append(scaled)

        transformed = []
        for out in output:
            h, w, _ = out.shape
            sliced = out[int(h * 0.2):, int(w * 0.2):]
            scaled = cv2.resize(sliced, CLASSIFIER_INPUT_SHAPE, interpolation=cv2.INTER_LINEAR)
            Logger.log_img(scaled)
            transformed.append(scaled)
        self.output = np.asarray(transformed)
        return self.output

    @staticmethod
    def is_shape_zero(shape):
        for axis in shape:
            if axis == 0:
                return True
        return False


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

        inp = X[-1].transpose((2, 0, 1)) / 255.
        prediction = self.model.predict(np.array([inp]))[-1]
        Logger.log('img', inp)
        Logger.log('prediction', prediction)
        self.output = prediction
        self.clear_cache()
        return self.output

    def load_model(self):
        # with open(os.path.join(GESTURE_PREDICTION_FOLDER, ARCHITECTURE_JSON_NAME)) as f:
        #     model = model_from_json(f.read(),
        #                             custom_objects={'f1': f1})
        # model.load_weights(os.path.join(GESTURE_PREDICTION_FOLDER, WEIGHTS_H5_NAME))
        model = load_model(os.path.join(GESTURE_PREDICTION_FOLDER, WEIGHTS_H5_NAME),
                           custom_objects={'f1': f1})
        return model
        # return load_model(os.path.join(GESTURE_PREDICTION_FOLDER, '90per.h5'))
        # return GestureClassifierMock.model()


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

        char_predictor = CNNTransformer.transformers[self.indices_of_transformers_to_combine[1]]
        gesture_prediction = np.insert(gesture_prediction, 0, [0.0])
        prediction = gesture_prediction * 0.8 + char_prediction * 0.2
        predicted_arg = np.argmax(prediction)
        char_predictor.add_to_context(int(predicted_arg))
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

        char_predictor = CNNTransformer.transformers[self.indices_of_transformers_to_combine[1]]
        char_predictor.add_to_previous_predictions(prediction[0][0])

    def load_model(self):
        return PredictionSelectionMock.model()
