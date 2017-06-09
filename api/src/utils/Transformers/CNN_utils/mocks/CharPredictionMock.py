from keras.models import Sequential, Model
from keras.layers import GRU, merge, Input, Dense


def model(time_steps=5, feature_length=37):
    rnn_size = 128
    inp = Input(shape=(time_steps, feature_length))
    gru_1 = GRU(rnn_size)(inp)
    prediction = Dense(feature_length, activation='softmax')(gru_1)
    m = Model(input=inp, output=prediction)
    return m
