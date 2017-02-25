from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from system import mkdir

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
