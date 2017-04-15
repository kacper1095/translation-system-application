from .Transformer import Transformer

import imutils
import cv2


class Resize(Transformer):
    out_key = 'original_img'

    def __init__(self, height=None, width=None):
        self.height = height
        self.width = width

    def transform(self, X, **transform_params):
        if type(X) == list:
            for i, img in enumerate(X):
                X[i] = cv2.resize(X[i], (self.width, self.height))
        elif len(X.shape) == 3:
            return cv2.resize(X, (self.width, self.height))
        else:
            raise ValueError('Unknown shape of data')
        return X

