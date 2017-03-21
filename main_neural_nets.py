import random

import numpy as np

from utils import logger
from utils import matlab

logger.log_to_files('nn')

# seed = int(random.random() * 1e6)
seed = 42
random.seed(seed)
np.random.seed(seed)
print("Seed =", seed)

from keras.utils.np_utils import to_categorical

import loader
import preprocessing
import keras_helper as helper
import feature_extractor
import validation

from models import *


def create_training_set(X, Y):
    return feature_extractor.extract_heartbeats(X, Y)


FREQUENCY = 300  # 300 points per second
WINDOW_SIZE = int(3 * FREQUENCY)
STEP = int(0.3 * FREQUENCY)
(X, Y) = loader.load_all_data()
X = preprocessing.normalize(X)
(Y, mapping) = preprocessing.format_labels(Y)
print('Input length', len(X))
print('Categories mapping', mapping)
(X, Y) = create_training_set(X, Y)
(X, Y) = preprocessing.shuffle_data(X, Y)
print('Training shape', X.shape)

# This is required for FCN
X = X.reshape((X.shape[0], 1, X.shape[1]))
print(X.shape)

impl = FCN(input_shape=X.shape[1:])
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

Xt, Xv, Yt, Yv = helper.train_test_split(subX, subY, 0.33)

Yt = to_categorical(Yt, len(mapping.keys()))
Yv = to_categorical(Yv, len(mapping.keys()))

model_saver = helper.best_model_saver(impl.name())
learning_optimizer = helper.model_learning_optimizer()
learning_stopper = helper.learning_stopper()
model.fit(Xt, Yt,
          nb_epoch=50,
          validation_data=(Xv, Yv),
          callbacks=[
              model_saver,
              learning_optimizer,
              learning_stopper
          ])

model.load_weights(model_saver.filepath)

validation.print_categorical_validation(Yv, model.predict(Xv), categorical=True)
