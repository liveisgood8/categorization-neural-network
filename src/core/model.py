import keras
import src.core.config as config

from keras.utils import plot_model
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D


def make_cnn_model(num_classes: int, input_shape: tuple) -> Sequential:
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    return model


def plot_model_to_file(model: Sequential) -> None:
    plot_model(model, to_file=config.MODEL_PLOT_PNG_FILENAME, show_shapes=True)


def save_model_weights(model: Sequential) -> None:
    model.save_weights(config.MODEL_WEIGHTS_FILENAME)


def load_model_weights(model: Sequential) -> Sequential:
    model.load_weights(config.MODEL_WEIGHTS_FILENAME)
    return model


def save_model(model: Sequential) -> None:
    save_model_weights(model)
    plot_model_to_file(model)
    model_json = model.to_json()
    with open(config.MODEL_JSON_FILENAME, 'w') as json_file:
        json_file.write(model_json)


def load_model() -> Sequential:
    with open(config.MODEL_JSON_FILENAME, 'r') as json_file:
        loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(config.MODEL_WEIGHTS_FILENAME)
    loaded_model.compile(loss=keras.losses.categorical_crossentropy,
                          optimizer=keras.optimizers.Adadelta(),
                          metrics=['accuracy'])
    return loaded_model
