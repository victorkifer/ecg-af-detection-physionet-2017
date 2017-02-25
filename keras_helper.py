from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from system import mkdir
import math

from sklearn.model_selection import train_test_split as dataset_split


def train_test_split(X, Y, split=0.33):
    return dataset_split(X, Y, test_size=split)


def model_saver(model_name):
    mkdir('outputs/models/' + model_name)
    return ModelCheckpoint('outputs/models/' + model_name + '/model_{epoch:02d}.hdf5',
                           monitor='val_loss',
                           verbose=1,
                           save_best_only=False,
                           save_weights_only=False,
                           mode='auto',
                           period=1)


def model_learning_optimizer():
    return ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                             patience=2, min_lr=0.1, verbose=1)


def learning_stopper():
    return EarlyStopping(monitor='val_loss', min_delta=0.01, patience=5, verbose=1)


def get_class_weights(Y, mu=0.8):
    """
    :param Y: labels
    :param mu: parameter to tune
    :return: class weights dictionary
    """
    train_categories_dist = dict()
    labels = set(Y)
    for label in labels:
        train_occurancies = sum([1 if label == y else 0 for y in Y])
        train_categories_dist[label] = train_occurancies

    total = sum(train_categories_dist.values())
    keys = train_categories_dist.keys()
    class_weight = dict()

    for key in keys:
        score = math.log(mu * total / float(train_categories_dist[key]))
        class_weight[key] = score if score > 1.0 else 1.0

    return class_weight
