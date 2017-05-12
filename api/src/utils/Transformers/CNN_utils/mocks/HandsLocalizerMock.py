from keras.models import Sequential
from keras.layers import Convolution2D


def model(img_channels=3, img_rows=240, img_cols=320):
    m = Sequential()
    m.add(Convolution2D(1, 1, 1, activation='sigmoid', input_shape=(img_channels, img_rows, img_cols)))
    return m