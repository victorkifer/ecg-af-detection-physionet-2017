import math
import numpy as np
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from utils.system import mkdir


def model_saver(model_name):
    mkdir('outputs/models/' + model_name)
    return ModelCheckpoint('outputs/models/' + model_name + '/model_{epoch:02d}.h5',
                           monitor='val_loss',
                           verbose=1,
                           save_best_only=False,
                           save_weights_only=False,
                           mode='auto',
                           period=1)


def best_model_saver(model_name):
    mkdir('outputs/models/' + model_name)
    return ModelCheckpoint('outputs/models/' + model_name + '/weights.best.h5',
                           monitor='val_loss',
                           verbose=1,
                           save_best_only=True,
                           save_weights_only=False,
                           mode='auto',
                           period=1)


def model_learning_optimizer():
    return ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                             patience=2, min_lr=0.01, verbose=1)


def learning_stopper():
    return EarlyStopping(monitor='val_loss', min_delta=0.01, patience=5, verbose=1)



