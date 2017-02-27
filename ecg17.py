import random
import numpy as np

# seed = int(random.random() * 1e6)
seed = 42
random.seed(seed)
np.random.seed(seed)
print("Seed =", seed)

from keras.utils.np_utils import to_categorical

import loader
import preprocessing
import keras_helper as helper
import validation

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
print('Input length', len(X))
print('Categories mapping', mapping)
(X, Y) = create_training_set(X, Y, WINDOW_SIZE, step=10, fadein=500)
(X, Y) = preprocessing.shuffle_data(X, Y)
print('Training shape', X.shape)

# This is required for FCN
X = X.reshape((X.shape[0], 1, X.shape[1]))
print(X.shape)

impl = ResNet(input_shape=X.shape[1:])
model = impl.model
model.summary()

NB_SAMPLES = 50000

subX = X[:NB_SAMPLES]
subY = Y[:NB_SAMPLES]

from collections import Counter

print("Distribution of categories before balancing")
counter = Counter(subY)
for key in counter.keys():
    print(key, counter[key])

# class_weights = helper.get_class_weights(subY)
class_weights = {
    0: 3,
    1: 1,
    2: 2,
    3: 30
}
print('Class weights', class_weights)

Xt, Xv, Yt, Yv = helper.train_test_split(subX, subY, 0.33)

(Xt, Yt) = preprocessing.balance_data(Xt, Yt, class_weights)

Yt = to_categorical(Yt, len(mapping.keys()))
Yv = to_categorical(Yv, len(mapping.keys()))

model.fit(Xt, Yt,
          nb_epoch=50,
          validation_data=(Xv, Yv),
          callbacks=[
              helper.model_saver(impl.name()),
              helper.model_learning_optimizer(),
              helper.learning_stopper()
          ])

validation.print_categorical_validation(model, Xv, Yv, mapping)
