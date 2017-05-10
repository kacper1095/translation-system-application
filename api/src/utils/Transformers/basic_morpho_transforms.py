from .Transformer import Transformer

import cv2


class Resizer(Transformer):
    def __init__(self, height=None, width=None):
        super(Resizer, self).__init__()
        self.height = height
        self.width = width
        self.out_key = 'resize'

    def transform(self, X, **transform_params):
        if type(X) == list:
            result = []
            for i, img in enumerate(X):
                result.append(cv2.resize(X[i], (self.width, self.height)))
        elif len(X.shape) == 3:
            self.output = cv2.resize(X, (self.width, self.height))
            return self.output
        else:
            raise ValueError('Unknown shape of data')
        self.output = result
        return self.output

