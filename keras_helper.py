from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from system import mkdir


def model_saver(model_name):
    mkdir('outputs/models/' + model_name)
    return ModelCheckpoint('outputs/models/' + model_name + '/model_{epoch:02d}.hdf5',
                           monitor='val_loss',
                           verbose=0,
                           save_best_only=False,
                           save_weights_only=False,
                           mode='auto',
                           period=1)


def model_learning_optimizer():
    return ReduceLROnPlateau(monitor='loss', factor=0.2,
                             patience=3, min_lr=0.1)
