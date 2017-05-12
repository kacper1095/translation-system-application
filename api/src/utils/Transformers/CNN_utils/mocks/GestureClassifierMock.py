from keras.models import Sequential
from keras.layers import GRU, Convolution3D, Flatten, Dense


def model(time_steps=5, img_channels=1, img_rows=64, img_cols=64):
    m = Sequential()
    m.add(Convolution3D(32, 5, 5, 5, activation='relu', input_shape=(img_channels, time_steps,
                                                                     img_rows, img_cols)))
    m.add(Convolution3D(64, 1, 3, 3, activation='relu'))
    m.add(Flatten())
    m.add(Dense(37, activation='softmax'))
    return m