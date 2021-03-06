import numpy as np
from typing import Dict, List


def spit(data_set_dict: Dict[int, List[np.ndarray]], train_size: int) -> (Dict[int, list], Dict[int, list]):
    train_data_set = {}
    test_data_set = {}

    for key in data_set_dict.keys():
        if len(data_set_dict[key]) < train_size:
            raise RuntimeError('train size must be smaller then data set size')
        train_data_set[key] = data_set_dict[key][:train_size]
        test_data_set[key] = data_set_dict[key][train_size:]

    return train_data_set, test_data_set
