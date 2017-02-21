from keras.callbacks import ModelCheckpoint
from system import mkdir


def model_saver(model_name):
    mkdir('outputs/models/')
    return ModelCheckpoint('outputs/models/' + model_name + '_{epoch:02d}.hdf5',
                                  monitor='val_loss',
                                  verbose=0,
                                  save_best_only=False,
                                  save_weights_only=False,
                                  mode='auto',
                                  period=1)