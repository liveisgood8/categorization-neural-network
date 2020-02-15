from src.core.parser.csv_dataset_parser import parse as csv_data_parse
from src.core.splitters.split_dict import spit as split_dataset


data_set = csv_data_parse('/media/Shared/Учеба/МКурс1_Семестр2/МашинноеОбучение/datasets/'
                                             'handwritten-mongolian-cyrillic-characters-database/HMCC balanced.csv')

train_data_set, test_data_set = split_dataset(data_set, 4000)

print(len(train_data_set[0]), len(test_data_set[0]))