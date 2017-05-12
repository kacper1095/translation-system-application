from .Transformer import Transformer
from src import common

import cv2
import numpy as np


class Resizer(Transformer):
    def __init__(self, height=None, width=None):
        super(Resizer, self).__init__()
        self.height = height
        self.width = width
        self.out_key = 'resize'

    def transform(self, X, **transform_params):
        if type(X) == list or len(X.shape) == 4:
            result = []
            for i, img in enumerate(X):
                result.append(cv2.resize(X[i], (self.width, self.height)))
        elif len(X.shape) == 3:
            self.output = cv2.resize(X, (self.width, self.height))
            return self.output
        else:
            raise ValueError('Unknown shape of data')
        self.output = np.asarray(result)
        return self.output


class Normalizer(Transformer):
    def __init__(self):
        super(Normalizer, self).__init__()
        self.out_key = 'normalized'

    def transform(self, X, **transform_params):
        X[:, 0, :, :] -= common.MEAN_VALUE_VGG[-1]
        X[:, 1, :, :] -= common.MEAN_VALUE_VGG[-2]
        X[:, 2, :, :] -= common.MEAN_VALUE_VGG[-3]
        return X


class BoxHands(Transformer):
    def __init__(self):
        super(BoxHands, self).__init__()
        self.out_key = 'boxed'

    def transform(self, X, **transform_params):
        self.output = np.random.random((5, 1, 64, 64))
        return self.output