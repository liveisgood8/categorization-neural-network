import csv
import numpy as np
from typing import Dict, List
from PyQt5.QtGui import QImage, QColor

import src.core.config as config


def parse(path: str, num_classes: int) -> Dict[int, List[np.ndarray]]:
    with open(path, newline='\n') as csv_file:
        data_set = prepare_data_set_dict(num_classes)
        data_reader = csv.reader(csv_file, delimiter=',')
        i = 0
        for row in data_reader:
            set_label = int(row[0])

            np_set = np.asarray(row[1:], dtype=np.float32)
            np_set = np.reshape(np_set, (-1, 28))

            data_set[set_label].append(np_set)

            i += 1
            if i % 1000 == 0:
                print('csv string parsed: ', i)

    return data_set


def prepare_data_set_dict(num_classes: int) -> Dict[int, list]:
    data_set_dict = {}
    for i in range(0, num_classes):
        data_set_dict[i] = []
    return data_set_dict


def dump_image(matrix: np.ndarray, number: int) -> None:
    image = QImage(28, 28, QImage.Format_RGB32)
    for i in range(0, matrix.shape[0]):
        for j in range(0, matrix.shape[1]):
            grayscale = matrix[i][j]
            image.setPixelColor(i, j, QColor(grayscale, grayscale, grayscale))
    image.save('dump/img' + str(number) + '.png')


