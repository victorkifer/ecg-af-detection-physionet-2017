from keras.engine import Input
from keras.engine import Model
from keras.layers import Activation, GlobalAveragePooling1D, Dropout, Dense, Reshape, Conv1D, merge, GlobalMaxPooling1D, \
    Convolution1D, MaxPooling1D, LSTM
from keras.layers import BatchNormalization
from keras.layers import Conv1D
from keras.layers.merge import add
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


class RCN(__BaseModel__):
    model = None

    def __init__(self, input_shape):
        inputs = Input(shape=input_shape)

        outputs = Reshape((1,) + input_shape)(inputs)

        outputs = LSTM(64, return_sequences=True)(outputs)
        outputs = LSTM(128, return_sequences=True)(outputs)
        outputs = LSTM(256, return_sequences=True)(outputs)
        outputs = LSTM(512, return_sequences=True)(outputs)
        outputs = Dropout(0.3)(outputs)
        outputs = LSTM(64, return_sequences=True)(outputs)
        outputs = LSTM(128)(outputs)
        outputs = Dropout(0.3)(outputs)

        outputs = Dense(256)(outputs)
        outputs = Dropout(0.2)(outputs)
        outputs = Dense(4)(outputs)
        outputs = Activation('softmax')(outputs)

        m = Model(inputs=inputs, output=outputs)
        m.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
        self.model = m


class FCN(__BaseModel__):
    model = None

    def __init__(self, input_shape):
        inputs = Input(shape=input_shape)
        outputs = Reshape((1,) + input_shape)(inputs)

        outputs = Conv1D(filters=256, kernel_size=32, padding="same")(outputs)
        outputs = BatchNormalization()(outputs)
        outputs = Activation('relu')(outputs)

        outputs = Conv1D(filters=512, kernel_size=16, padding="same")(outputs)
        outputs = BatchNormalization()(outputs)
        outputs = Activation('relu')(outputs)

        outputs = GlobalMaxPooling1D()(outputs)
        outputs = Reshape((1, 512))(outputs)

        outputs = Conv1D(filters=256, kernel_size=8, padding="same")(outputs)
        outputs = BatchNormalization()(outputs)
        outputs = Activation('relu')(outputs)

        outputs = GlobalMaxPooling1D()(outputs)
        outputs = Dense(128)(outputs)
        outputs = Activation('relu')(outputs)
        outputs = Dropout(0.75)(outputs)
        outputs = Dense(4)(outputs)
        outputs = Activation('softmax')(outputs)

        m = Model(inputs=inputs, output=outputs)
        m.compile(optimizer='adagrad',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
        self.model = m


class ResNet(__BaseModel__):
    model = None

    def __init__(self, input_shape):
        inputs = Input(shape=input_shape)
        out = Reshape((1,) + input_shape)(inputs)
        out = ResNet.__conv_bn_relu(out, input_shape, 64, 8)
        out = ResNet.__conv_bn_relu(out, input_shape, 128, 5)
        out = ResNet.__conv_bn_relu(out, input_shape, 128, 3)
        out = GlobalAveragePooling1D()(out)
        out = Dense(4)(out)
        out = Activation('softmax')(out)
        model = Model(inputs=inputs, outputs=out)
        model.compile(optimizer='adagrad',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        self.model = model

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
