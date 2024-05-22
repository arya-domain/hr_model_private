import tensorflow as tf
from keras.models import Model
from keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D,  BatchNormalization, GlobalMaxPooling1D, Input, Permute
from keras.layers import Multiply
from keras.optimizers import Adam

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

def lstm_block1(x, units):
    x = Permute((2, 1))(x)
    x = LSTM(units, return_sequences=True)(x)
    x = Dropout(0.2)(x)
    x = Dense(units, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)

    return x

def lstm_block2(x, units):
    x = LSTM(units, return_sequences=True)(x)
    x = Dense(units, activation="relu")(x)
    x = Permute((2, 1))(x)
    return x

def conv_block(x, filters, kernel_size):
    x = Conv1D(filters, kernel_size, activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    return x


def classifier():
    # Define the model
    input_layer = Input(shape=(408, 1))

    # block 1
    x = lstm_block2(input_layer, 256)
    x = conv_block(x, 64, 3)
    x = lstm_block1(x, 256)

    # block 2
    x1 = conv_block(input_layer, 64, 3)
    x1 = lstm_block2(x, 512)
    x1 = lstm_block2(x, 256)
    x1 = Permute((2, 1))(x1)

    x2 = Multiply()([x, x1])

    # block 2
    x1 = conv_block(x2, 64, 3)
    x1 = lstm_block2(x, 256)
    x1 = lstm_block2(x, 128)
    x1 = Permute((2, 1))(x1)

    # block 1
    x = lstm_block2(x2, 128)
    x = conv_block(x, 64, 3)
    x = lstm_block1(x, 128)

    x2 = Multiply()([x, x1])

    x = Dense(16, activation="relu")(x2)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)

    output_layer = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=input_layer, outputs=output_layer)

    # Compile the model
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    return model