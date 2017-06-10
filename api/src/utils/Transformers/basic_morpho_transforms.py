from .Transformer import Transformer
from src import common
from src.utils.Logger import Logger

import cv2
import numpy as np
import imutils


class Resizer(Transformer):
    def __init__(self, height=None, width=None):
        super(Resizer, self).__init__()
        self.height = height
        self.width = width
        self.out_key = 'resize'

    def transform(self, X, **transform_params):
        if len(X.shape) == 4:
            result = []
            for i, img in enumerate(X):
                result.append(imutils.resize(X[i], width=self.width, height=self.height))
        elif len(X.shape) == 3:
            self.output = imutils.resize(X, width=self.width, height=self.height)
            return self.output
        else:
            raise ValueError('Unknown shape of data')
        self.output = np.asarray(result)
        return self.output


class BGR2HSV(Transformer):
    def __init__(self):
        super(BGR2HSV, self).__init__()
        self.out_key = 'in_hsv_space'

    def transform(self, X, **transform_params):
        if type(X) == list or len(X.shape) == 4:
            result = []
            for i, img in enumerate(X):
                result.append(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        elif len(X.shape) == 3:
            self.output = cv2.cvtColor(X, cv2.COLOR_BGR2HSV)
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
        self.output = X.copy()/255.
        return self.output


class BoxHands(Transformer):
    def __init__(self):
        super(BoxHands, self).__init__()
        self.out_key = 'boxed'

    def transform(self, X, **transform_params):
        Logger.log("x_trans_shape", X.shape)
        if type(X) == list or len(X.shape) == 4:
            grayscaled = []
            for sample in X:
                img_gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)[:, :, np.newaxis]
                img_gray /= 255.
                grayscaled.append(img_gray)
            tensor = np.array(grayscaled)
            self.output = tensor
            return self.output
        elif len(X.shape) == 3:
            self.output = cv2.cvtColor(X, cv2.COLOR_BGR2GRAY)[:, :, np.newaxis]
            return self.output
        else:
            raise ValueError("Unknown shape of data")

