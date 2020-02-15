from typing import Dict


def spit(data_set_dict: Dict[int, list], train_size: int):
    train_data_set = {}
    test_data_set = {}

    for key in data_set_dict.keys():
        if len(data_set_dict[key]) < train_size:
            raise RuntimeError('train size must be smaller then data set size')
        train_data_set[key] = data_set_dict[key][:train_size]
        test_data_set[key] = data_set_dict[key][train_size:]

    return train_data_set, test_data_set
