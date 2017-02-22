from keras.engine import Input
from keras.engine import merge
from keras.layers import Activation, GlobalAveragePooling1D, Dropout, Dense, Flatten
from keras.layers import BatchNormalization
from keras.layers import Convolution1D
from keras.models import Sequential


class __BaseModel__():
    def name(self):
        return type(self).__name__.lower()


class MLP(__BaseModel__):
    model = None

    def __init__(self, input_shape):
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
        self.model = m


class FCN(__BaseModel__):
    model = None

    def __init__(self, input_shape):
        m = Sequential()
        m.add(Convolution1D(nb_filter=128, filter_length=8, input_shape=input_shape, border_mode="same"))
        m.add(BatchNormalization())
        m.add(Activation('relu'))
        m.add(Convolution1D(nb_filter=256, filter_length=5, border_mode="same"))
        m.add(BatchNormalization())
        m.add(Activation('relu'))
        m.add(Convolution1D(nb_filter=128, filter_length=3, border_mode="same"))
        m.add(BatchNormalization())
        m.add(Activation('relu'))
        m.add(GlobalAveragePooling1D())
        m.add(Dense(4))
        m.add(Activation('softmax'))
        m.compile(optimizer='adagrad',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
        self.model = m


class ResNet(__BaseModel__):
    model = None

    def __init__(self, input_shape):
        m = Sequential()
        inputs = Input(shape=input_shape)
        m.add(ResNet.__conv_bn_relu(inputs, input_shape, 64, 8))
        m.add(ResNet.__conv_bn_relu(inputs, input_shape, 128, 5))
        m.add(ResNet.__conv_bn_relu(inputs, input_shape, 128, 3))
        m.add(GlobalAveragePooling1D())
        m.add(Dense(4))
        m.add(Activation('softmax'))
        m.compile(optimizer='adagrad',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
        self.model = m

    @staticmethod
    def __conv_bn_relu(input_tensor, input_shape, nb_filter, filter_length):
        block = Convolution1D(nb_filter=nb_filter, filter_length=filter_length, input_shape=input_shape, border_mode="same")(input_tensor)
        block = BatchNormalization()(block)
        block = Activation('relu')(block)
        block = Convolution1D(nb_filter=nb_filter, filter_length=filter_length, border_mode="same")(block)
        block = BatchNormalization()(block)
        block = Activation('relu')(block)
        block = Convolution1D(nb_filter=nb_filter, filter_length=filter_length, border_mode="same")(block)
        block = BatchNormalization()(block)
        block = merge([block, input_tensor], 'sum')
        block = Activation('relu')(block)
        return block
