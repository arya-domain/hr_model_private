import tensorflow as tf
from keras.models import Model
from keras.optimizers import Adam
from imblearn.over_sampling import SMOTE
from keras.layers import *

def conv_block(x, filters, kernel_size):
    x = Conv1D(filters, kernel_size, activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    return x

def denseblock(x, units):
    x = Dense(units)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = tf.expand_dims(x, axis=-1)
    return x


def classifier():

    input_layer = Input(shape=(408,))

    x = denseblock(input_layer, 1024)
    x =conv_block(x, 2, 2)

    x = denseblock(input_layer, 512)
    x =conv_block(x, 2, 2)

    x = denseblock(x, 256)
    x =conv_block(x, 2, 2)
    # x = x * 0.2

    x = denseblock(x, 128)
    x =conv_block(x, 2, 2)
    # x = x * 0.2

    x = denseblock(x, 64)
    x =conv_block(x, 2, 2)

    x = Flatten()(x)

    x = Dense(32, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)

    x = Dense(16, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)

    output_layer = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=input_layer, outputs=output_layer)

    # Compile the model
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    return model