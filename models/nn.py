from keras.engine import Input
from keras.engine import Model
from keras.layers import Activation, GlobalAveragePooling1D, Dropout, Dense, Reshape, GlobalMaxPooling1D, GRU
from keras.layers import BatchNormalization
from keras.layers import Conv1D
from keras.layers.merge import add
from keras.models import Sequential
from keras.utils import to_categorical

from models.base import EcgModel
from models.neural_networks import keras_helper as helper
from preprocessing import categorizer
from utils.system import copy_file


class NeuralNetworkEcgModel(EcgModel):
    model = None
    model_file = "weights.h5"
    n_classes = len(categorizer.__MAPPING__.keys())

    def __init__(self, input_shape):
        self.model = self.create_model(input_shape)

    def restore(self):
        self.model.load_weights("weights.h5")

    def create_model(self, input_shape):
        raise NotImplementedError()

    def get_learning_optimizer(self):
        return helper.model_learning_optimizer()

    def get_learning_stopper(self):
        return helper.learning_stopper()

    @staticmethod
    def from_categorical(y):
        return [x.tolist().index(max(x)) for x in y]

    def fit(self, x, y, validation=None):
        callbacks = list()
        model_saver = helper.best_model_saver(self.name())
        callbacks.append(model_saver)

        learning_optimizer = self.get_learning_optimizer()
        if learning_optimizer is not None and validation is not None:
            callbacks.append(learning_optimizer)

        learning_stopper = self.get_learning_stopper()
        if learning_stopper is not None:
            callbacks.append(learning_stopper)

        y = to_categorical(y, num_classes=self.n_classes)

        if validation is not None:
            validation = (validation[0], to_categorical(validation[1], self.n_classes))

        self.model.fit(x, y,
                       epochs=50,
                       validation_data=validation,
                       callbacks=callbacks)
        self.model.load_weights(model_saver.filepath)
        copy_file(model_saver.filepath, "weights.h5")

    def predict(self, x):
        y_pred = self.model.predict(x)
        return NeuralNetworkEcgModel.from_categorical(y_pred)


class MlpEcgModel(NeuralNetworkEcgModel):
    def __init__(self, input_shape):
        super().__init__(input_shape)

    def create_model(self, input_shape):
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


class RecurrentEcgModel(NeuralNetworkEcgModel):
    def __init__(self, input_shape):
        super().__init__(input_shape)

    def create_model(self, input_shape):
        m = Sequential()
        m.add(Reshape((1,) + input_shape, input_shape=input_shape))
        m.add(GRU(64, return_sequences=True))
        m.add(GRU(32))
        m.add(Dense(16, activation='relu'))
        m.add(Dropout(0.5))
        m.add(Dense(4))
        m.add(Activation('softmax'))
        m.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
        return m

class FcnEcgModel(NeuralNetworkEcgModel):
    def __init__(self, input_shape):
        super().__init__(input_shape)

    def create_model(self, input_shape):
        inputs = Input(shape=input_shape)
        outputs = Reshape((1,) + input_shape)(inputs)

        outputs = Conv1D(filters=128, kernel_size=32, padding="same")(outputs)
        outputs = BatchNormalization()(outputs)
        outputs = Activation('relu')(outputs)

        outputs = Conv1D(filters=256, kernel_size=16, padding="same")(outputs)
        outputs = BatchNormalization()(outputs)
        outputs = Activation('relu')(outputs)

        outputs = Conv1D(filters=128, kernel_size=8, padding="same")(outputs)
        outputs = BatchNormalization()(outputs)
        outputs = Activation('relu')(outputs)

        outputs = GlobalMaxPooling1D()(outputs)
        outputs = Dense(128)(outputs)
        outputs = Activation('relu')(outputs)
        outputs = Dropout(0.75)(outputs)
        outputs = Dense(4)(outputs)
        outputs = Activation('softmax')(outputs)

        m = Model(inputs=inputs, outputs=outputs)
        m.compile(optimizer='adagrad',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
        return m


class ResNetEcgModel(NeuralNetworkEcgModel):
    def __init__(self, input_shape):
        super().__init__(input_shape)

    def create_model(self, input_shape):
        inputs = Input(shape=input_shape)
        out = Reshape((1,) + input_shape)(inputs)
        out = ResNetEcgModel.__conv_bn_relu(out, input_shape, 64, 23)
        out = ResNetEcgModel.__conv_bn_relu(out, input_shape, 128, 8)
        out = ResNetEcgModel.__conv_bn_relu(out, input_shape, 128, 3)
        out = GlobalAveragePooling1D()(out)
        out = Dense(4, activation='softmax')(out)
        m = Model(inputs=inputs, outputs=out)
        m.compile(optimizer='adagrad',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
        return m

    @staticmethod
    def __conv_bn_relu(input_tensor, input_shape, filters, kernel_size):
        block = Conv1D(filters=filters, kernel_size=kernel_size, input_shape=input_shape,
                       padding="same")(input_tensor)
        block = BatchNormalization()(block)
        block = Activation('relu')(block)
        block = Conv1D(filters=filters, kernel_size=kernel_size, padding="same")(block)
        block = BatchNormalization()(block)
        block = Activation('relu')(block)
        block = Conv1D(filters=filters, kernel_size=kernel_size, padding="same")(block)
        block = BatchNormalization()(block)
        is_expand_channels = not (input_shape[-1] == filters)
        if is_expand_channels:
            shortcut_y = Conv1D(filters, 1, padding='same')(input_tensor)
            shortcut_y = BatchNormalization()(shortcut_y)
        else:
            shortcut_y = BatchNormalization()(input_shape)

        block = add([block, shortcut_y])
        block = Activation('relu')(block)
        return block
