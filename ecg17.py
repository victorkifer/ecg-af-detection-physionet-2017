from keras.layers import Dense, Dropout, Activation
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import GlobalAveragePooling1D
from keras.utils.np_utils import to_categorical

import numpy as np
import matplotlib.pyplot as plt

import loader
import preprocessing
import keras_helper as helper


def plot_signal(signal):
    plt.clf()
    time = np.arange(0, len(signal))
    plt.scatter(time, signal, s=1)  #


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


def mlp(input_shape):
    m = Sequential()
    m.add(Dropout(0.1, input_shape=input_shape))
    m.add(Dense(500))
    m.add(Activation('relu'))
    m.add(Dropout(0.2))
    m.add(Dense(500))
    m.add(Activation('relu'))
    m.add(Dropout(0.2))
    m.add(Dense(500))
    m.add(Activation('relu'))
    m.add(Dropout(0.3))
    m.add(Dense(4))
    m.add(Activation('softmax'))
    m.compile(optimizer='adagrad',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    return m


def fcn(input_shape):
    m = Sequential()
    m.add(Dense(128, input_shape=input_shape))
    m.add(BatchNormalization())
    m.add(Activation('relu'))
    m.add(Dense(256))
    m.add(BatchNormalization())
    m.add(Activation('relu'))
    m.add(Dense(128))
    m.add(BatchNormalization())
    m.add(Activation('relu'))
    m.add(GlobalAveragePooling1D())
    m.add(Dense(4))
    m.add(Activation('softmax'))
    m.compile(optimizer='adagrad',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    return m


model = mlp(input_shape=(WINDOW_SIZE,))
model.summary()
Y_one_hot_vector = to_categorical(Y, len(mapping.keys()))
model.fit(X, Y_one_hot_vector, validation_split=0.33, callbacks=[helper.model_saver('mlp')])
