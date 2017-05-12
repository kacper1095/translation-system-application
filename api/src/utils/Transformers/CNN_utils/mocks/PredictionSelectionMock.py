from keras.layers import Convolution2D, Input
from keras.models import Model


def model(nb_of_letters=37):
    inp = Input(shape=(1, 2, nb_of_letters))
    convo = Convolution2D(nb_of_letters, 2, 1)(inp)
    convo = Convolution2D(1, 1, 1)(convo)
    m = Model(input=inp, output=convo)
    return m
