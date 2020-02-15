import numpy as np
import random
from keras import backend as keras_backend
from keras.utils import to_categorical
from typing import Dict, List

import src.core.config as config


def convert(data_set: Dict[int, List[np.ndarray]]) -> (np.ndarray, int):
    num_classes = len(data_set)
    data_array = []

    for key in data_set.keys():
        for j in range(0, len(data_set[key])):
            data_array.append({key: data_set[key][j]})

    random.shuffle(data_array)

    data_np_array_labels = np.empty(shape=(num_classes * len(data_set[0]), num_classes))
    data_np_array = np.empty(shape=(num_classes * len(data_set[0]), config.IMG_WIDTH, config.IMG_HEIGHT),
                             dtype=np.float32)
    for i, e in enumerate(data_array, start=0):
        class_label = list(e.keys())[0]
        data_np_array[i] = e[class_label]

        data_np_array_labels[i] = to_categorical(class_label, num_classes)

    if keras_backend.image_data_format() == 'channels_first':
        data_np_array = np.reshape(data_np_array, (data_np_array.shape[0], 1, config.IMG_WIDTH, config.IMG_HEIGHT))
    else:
        data_np_array = np.reshape(data_np_array, (data_np_array.shape[0], config.IMG_WIDTH, config.IMG_HEIGHT, 1))

    return data_np_array, data_np_array_labels
