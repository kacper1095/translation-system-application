from __future__ import print_function

from ..Transformer import Transformer
from ...AsciiEncoder import AsciiEncoder
from .mocks import GestureClassifierMock, CharPredictionMock, PredictionSelectionMock, HandsLocalizerMock

from keras.utils.np_utils import to_categorical
from keras.models import model_from_json
from src.common import (
    HANDS_SEGMENTATION_FOLDER, ARCHITECTURE_JSON_NAME, WEIGHTS_H5_NAME, CHAR_PREDICTION_FOLDER, WEIGHTS_HDF5_NAME
    )
from src.utils.Logger import Logger

import sys
import numpy as np
import os
import cv2


class CNNTransformer(Transformer):
    transformers = []

    def __init__(self):
        super(CNNTransformer, self).__init__()
        self.model = None

    def load_model(self):
        pass


class HandsLocalizer(CNNTransformer):
    def __init__(self):
        super(HandsLocalizer, self).__init__()
        self.model = self.load_model()
        self.out_key = 'hands'

    def transform(self, X, **transform_params):
        Logger.log("shape: " + str(X.max()))
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
        prediction = self.model.predict(np.asarray([X.transpose(1, 0, 2, 3)]))
        self.output = prediction[0]
        return self.output

    def load_model(self):
        return GestureClassifierMock.model()


class CharPredictor(CNNTransformer):
    __previous_predictions = to_categorical(np.random.randint(len(AsciiEncoder.AVAILABLE_CHARS), size=20),
                                            len(AsciiEncoder.AVAILABLE_CHARS))

    def __init__(self, num_of_chars):
        super(CharPredictor, self).__init__()
        self.num_of_chars = num_of_chars
        self.model = self.load_model()
        self.out_key = 'chars'

    def transform(self, X, **transform_params):
        x_data = CharPredictor.__previous_predictions
        self.output = self.model.predict(np.asarray([x_data]))[0]
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
        np.append(CharPredictor.__previous_predictions[1:],
                  to_categorical(prediction, len(AsciiEncoder.AVAILABLE_CHARS)), axis=0)

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
        previous_transformer_output = []
        for index in self.indices_of_transformers_to_combine:
            previous_transformer_output.append(CNNTransformer.transformers[index].output)
        x_data = np.asarray([[previous_transformer_output]])
        prediction = self.model.predict(x_data)
        predicted_arg = np.argmax(prediction[0])
        CharPredictor.add_to_previous_predictions(predicted_arg)
        self.output = prediction
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
