from sklearn.base import TransformerMixin

import numpy as np


class Transformer(TransformerMixin):
    required_keys = []
    out_key = None

    def __init__(self):
        self.output = None

    def score(self, X):
        return 0

    def transform(self, X, **transform_params):
        return X

    def get_name(self):
        return self.__class__.__name__

    def assign_data(self, args, data, kwargs):
        kwargs[self.out_key] = self.transform(*args, **data)
        return kwargs

    def transform_with_all(self, *args, **kwargs):
        data = {}
        for key in self.required_keys:
            if kwargs.has_key(key):
                data[key] = kwargs[key]
            else:
                print(key)
                return
        if not self.out_key:
            print(self.get_name())
            out = [self.transform(*args, **data)]
            out.extend(args)
            args = out
        else:
            print(self.get_name(), self.out_key)
            kwargs = self.assign_data(args, data, kwargs)
        return args, kwargs