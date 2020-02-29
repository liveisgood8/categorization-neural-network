import numpy as np
import sys
import matplotlib.pyplot as plt
import src.core.config as config

from keras import backend as keras_backend
from keras.utils import to_categorical
from emnist import extract_training_samples
from emnist import extract_test_samples
from src.app.main_application import start as start_application
from src.core.model import make_cnn_model
from src.core.model import save_model
from src.core.loaders import tf_loader


num_classes = 27
batch_size = 128
epochs = 8


def train_and_save_mode(show_history=False):
    print('Train started, show history:', show_history)

    x_train, y_train = extract_training_samples('letters')
    x_test, y_test = extract_test_samples('letters')

    x_train = np.copy(x_train).astype(np.float32)
    x_test = np.copy(x_test).astype(np.float32)

    x_train /= 255
    x_test /= 255

    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    if keras_backend.image_data_format() == 'channels_first':
        x_train = np.reshape(x_train, (x_train.shape[0], 1, config.IMG_WIDTH, config.IMG_HEIGHT))
        x_test = np.reshape(x_test, (x_test.shape[0], 1, config.IMG_WIDTH, config.IMG_HEIGHT))
        input_shape = (1, config.IMG_WIDTH, config.IMG_HEIGHT)
    else:
        x_train = np.reshape(x_train, (x_train.shape[0], config.IMG_WIDTH, config.IMG_HEIGHT, 1))
        x_test = np.reshape(x_test, (x_test.shape[0], config.IMG_WIDTH, config.IMG_HEIGHT, 1))
        input_shape = (config.IMG_WIDTH, config.IMG_HEIGHT, 1)

    print('Train input shape:', x_train.shape)
    print('Train output shape:', y_train.shape)
    print('Test input shape:', x_test.shape)
    print('Test output shape:', y_test.shape)

    model = make_cnn_model(config.IMG_WIDTH, config.IMG_HEIGHT, num_classes, input_shape)
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, y_test))

    save_model(model)

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    if show_history:
        # summarize history for accuracy
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()


def main():
    tf_loader.load()
    if '--train' in sys.argv:
        show_history = '--train-history' in sys.argv
        train_and_save_mode(show_history)
    start_application()


if __name__ == '__main__':
    main()
