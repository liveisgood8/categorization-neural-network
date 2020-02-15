from src.core.parser.csv_dataset_parser import parse as csv_data_parse
from src.core.preparers.split_dict import spit as split_dataset
from src.core.preparers.dict_set_to_learn_set import convert as convert_to_learn_set

num_classes = 65
data_set_file_path = '/home/nexus/HMCC balanced.csv'
# data_set_file_path = '/media/Shared/Учеба/МКурс1_Семестр2/МашинноеОбучение/datasets/' \
#                      'handwritten-mongolian-cyrillic-characters-database/HMCC balanced.csv'

data_set = csv_data_parse(data_set_file_path, num_classes)

train_data_set, test_data_set = split_dataset(data_set, 4000)
x_train, y_train = convert_to_learn_set(train_data_set)
x_test, y_test = convert_to_learn_set(test_data_set)

