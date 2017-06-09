from keras.models import Model
from keras.layers import Convolution2D, Input, MaxPooling2D, UpSampling2D, merge


def model(img_channels=3, img_rows=240, img_cols=320):
    inp = Input(shape=(img_channels, img_rows, img_cols))
    convo = Convolution2D(12, 7, 7, init='he_normal', activation='relu', border_mode='same')(inp)
    pool = MaxPooling2D((4, 4))(convo)
    convo = Convolution2D(16, 7, 7, init='he_normal', activation='relu', border_mode='same')(pool)
    convo_2 = Convolution2D(16, 1, 1, init='he_normal', activation='relu', border_mode='same')(convo)
    m = merge([convo, convo_2], mode='sum')
    unpool = UpSampling2D((4, 4))(m)
    convo = Convolution2D(12, 7, 7, init='he_normal', activation='relu', border_mode='same')(unpool)
    final = Convolution2D(1, 1, 1, init='he_normal', activation='sigmoid', border_mode='same')(convo)
    model = Model(input=inp, output=final)
    return model