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


def print_categorical_validation(model, valX, valY, mapping):
    correct = [x.tolist().index(max(x)) for x in valY]
    predicted = [x.tolist().index(max(x)) for x in model.predict(valX)]

    values = [correct[i] == predicted[i] for i in range(len(correct))]
    accuracy = values.count(True) * 1.0 / len(correct)
    
    matrix_size = len(mapping.keys())
    
    import numpy as np
    val = np.zeros((matrix_size, matrix_size), np.int32)
    for i in range(len(correct)):
        c = correct[i]
        p = predicted[i]
        val[c][p] += 1;
        
    print('Overal accuracy', accuracy)
    for i in range(matrix_size):
        classified = val[i][i]
        total = max(sum(val[i]), 1)
        print(mapping[i], 'accuracy is', classified / total)


FREQUENCY = 300  # 300 points per second
WINDOW_SIZE = 1 * FREQUENCY
STEP = int(0.2 * FREQUENCY)
(X, Y) = loader.load_all_data()
(Y, mapping) = preprocessing.format_labels(Y)
print('Input length', len(X))
print('Categories mapping', mapping)
(X, Y) = create_training_set(X, Y, WINDOW_SIZE, 10)
print('Training shape', X.shape)

# This is required for FCN
X = X.reshape((X.shape[0], 1, X.shape[1]))
print(X.shape)

impl = ResNet(input_shape=X.shape[1:])
model = impl.model
model.summary()


NB_SAMPLES = 100000

subX = X[:NB_SAMPLES]
subY = Y[:NB_SAMPLES]

from collections import Counter
counter = Counter(subY)
for key in counter.keys():
    print(key, counter[key])

subY = to_categorical(subY, len(mapping.keys()))

Xt, Xv, Yt, Yv = helper.train_test_split(subX, subY, 0.33)

model.fit(Xt, Yt,
          nb_epoch=50,
          validation_data=(Xv, Yv),
          callbacks=[
                  helper.model_saver(impl.name()),
                  helper.model_learning_optimizer(),
                  helper.learning_stopper()
                  ])

print_categorical_validation(model, Xv, Yv, mapping)
