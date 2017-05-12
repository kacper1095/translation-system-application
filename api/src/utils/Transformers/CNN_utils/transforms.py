from __future__ import print_function

from ..Transformer import Transformer
from ...AsciiEncoder import AsciiEncoder
from .mocks import GestureClassifierMock, CharPredictionMock, PredictionSelectionMock

from keras.utils.np_utils import to_categorical
from keras.models import model_from_json
from src.common import HANDS_SEGMENTATION_FOLDER, ARCHITECTURE_JSON_NAME, WEIGHTS_H5_NAME

import sys
import numpy as np
import os


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
        last_frame = np.asarray([X.transpose(0, 3, 1, 2)[0]])
        prediction = self.model.predict(last_frame)
        output = X.copy().transpose(0, 3, 1, 2)
        output[-1] = prediction[0] * 255
        self.output = output
        return self.output

    def load_model(self):
        with open(os.path.join(HANDS_SEGMENTATION_FOLDER, ARCHITECTURE_JSON_NAME), 'r') as f:
            model = model_from_json(f.read())
        model.load_weights(os.path.join(HANDS_SEGMENTATION_FOLDER, WEIGHTS_H5_NAME))
        return model


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
    __previous_predictions = []

    def __init__(self, num_of_chars):
        super(CharPredictor, self).__init__()
        self.num_of_chars = num_of_chars
        self.model = self.load_model()
        self.out_key = 'chars'

    def transform(self, X, **transform_params):
        x_data = self.prepare_data()
        self.output = self.model.predict(np.asarray([x_data]))[0]
        return self.output

    def prepare_data(self):
        predictions = CharPredictor.__previous_predictions
        if len(predictions) >= self.num_of_chars:
            while len(predictions) > self.num_of_chars:
                predictions.pop(0)
        else:
            while len(predictions) != self.num_of_chars:
                predictions.insert(0, 0)
        return to_categorical(predictions, len(AsciiEncoder.AVAILABLE_CHARS))

    @staticmethod
    def add_to_previous_predictions(prediction):
        if prediction.shape != (1,):
            prediction = np.argmax(prediction)
        CharPredictor.__previous_predictions.append(prediction)

    def load_model(self):
        return CharPredictionMock.model(time_steps=self.num_of_chars, feature_length=len(AsciiEncoder.AVAILABLE_CHARS))


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
        CharPredictor.add_to_previous_predictions(prediction[0][0])
        self.output = prediction[0][0]
        return self.output

    def fit_transform(self, X, y=None, **fit_params):
        previous_transformer_output = []
        for index in self.indices_of_transformers_to_combine:
            print(CNNTransformer.transformers[index].output, file=sys.stderr)
            previous_transformer_output.append(CNNTransformer.transformers[index].output)
        x_data = np.asarray([[previous_transformer_output]])
        prediction = self.model.fit(x_data, y)
        CharPredictor.add_to_previous_predictions(prediction[0][0])

    def load_model(self):
        return PredictionSelectionMock.model()
