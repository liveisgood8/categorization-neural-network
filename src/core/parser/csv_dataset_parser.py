import csv
import numpy as np


def parse(path: str, num_classes: int):
    with open(path, newline='\n') as csv_file:
        data_set = prepare_dataset_dict(num_classes)
        data_reader = csv.reader(csv_file, delimiter=',')
        i = 0
        for row in data_reader:
            set_label = int(row[0])

            np_set = np.asarray(row[1:], dtype=np.int)
            np_set = np.reshape(np_set, (-1, 28))

            data_set[set_label].append(np_set)

            i += 1
            if i % 1000 == 0:
                print('csv parser readed strings: ', i)

    return data_set


def prepare_dataset_dict(num_classes: int):
    data_set_dict = {}
    for i in range(0, num_classes + 1):
        data_set_dict[i] = []
    return data_set_dict

