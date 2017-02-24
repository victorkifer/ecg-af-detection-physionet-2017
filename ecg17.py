from keras.layers import Dense, Dropout, Activation
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import GlobalAveragePooling1D
from keras.utils.np_utils import to_categorical

import numpy as np

import loader
import preprocessing
import keras_helper as helper

from models import *


def create_training_set(X, Y, window_size, step, fadein=0, fadeout=0):
    x_out = []
    y_out = []
    for i in range(len(X)):
        x = X[i]
        y = Y[i]
        out = preprocessing.convolution(x, window_size, step, fadein, fadeout)
        for o in out:
            x_out.append(o)
            y_out.append(y)
    return (np.array(x_out), y_out)


FREQUENCY = 300  # 300 points per second
WINDOW_SIZE = 1 * FREQUENCY
STEP = int(0.2 * FREQUENCY)
(X, Y) = loader.load_all_data()
(Y, mapping) = preprocessing.format_labels(Y)
print('Input shape', len(X), len(Y))
print('Categories mapping', mapping)
(X, Y) = create_training_set(X, Y, WINDOW_SIZE, 10)
print('Training shape', len(X), len(Y))

# This is required for FCN
X = X.reshape((X.shape[0], 1, X.shape[1]))
print(X.shape)

impl = ResNet(input_shape=X.shape[1:])
model = impl.model
model.summary()
Y_one_hot_vector = to_categorical(Y, len(mapping.keys()))
model.fit(X[:500000], Y_one_hot_vector[:500000], validation_split=0.33,
          callbacks=[helper.model_saver(impl.name())])
