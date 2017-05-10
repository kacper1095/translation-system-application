from ..Transformer import Transformer

import numpy as np


class CNNTransformer(Transformer):
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
        return np.random.normal(0, 1, size=(len(X),) + X[0].shape) * 255

    def load_model(self):
        return 0


class GestureClassifier(CNNTransformer):
    def __init__(self):
        super(GestureClassifier, self).__init__()
        self.model = self.load_model()
        self.out_key = 'gesture'

    def transform(self, X, **transform_params):
        return np.random.normal(0, 1, size=(36,))

    def load_model(self):
        return 0


class CharPredictor(CNNTransformer):
    def __init__(self, num_of_chars):
        super(CharPredictor, self).__init__()
        self.model = self.load_model()
        self.num_of_chars = num_of_chars
        self.out_key = 'chars'

    def transform(self, X, **transform_params):
        return np.random.normal(0, 1, size=(36,))

    def load_model(self):
        return 0


class PredictionSelector(CNNTransformer):
    def __init__(self, indices_of_transformers_to_combine):
        super(PredictionSelector, self).__init__()
        self.model = self.load_model()
        self.indices_of_transformers_to_combine = indices_of_transformers_to_combine
        self.out_key = 'prediction_selection'

    def transform(self, X, **transform_params):
        return np.random.normal(0, 1, size=(36,))

    def load_model(self):
        return 0
