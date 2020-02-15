import src.core.config as config

from keras import backend as keras_backend
from src.core.parser.csv_dataset_parser import parse as csv_data_parse
from src.core.preparers.split_dict import spit as split_dataset
from src.core.preparers.dict_set_to_learn_set import convert as convert_to_learn_set
from src.core.model import make_cnn_model
from src.core.model import save_model
from src.core.loaders import tf_loader

num_classes = 66
batch_size = 128
epochs = 12
data_set_file_path = '/home/nexus/HMCC balanced.csv'
# data_set_file_path = '/media/Shared/Учеба/МКурс1_Семестр2/МашинноеОбучение/datasets/' \
#                      'handwritten-mongolian-cyrillic-characters-database/HMCC balanced.csv'

tf_loader.load()

data_set = csv_data_parse(data_set_file_path, num_classes)

train_data_set, test_data_set = split_dataset(data_set, 4000)
x_train, y_train = convert_to_learn_set(train_data_set)
x_test, y_test = convert_to_learn_set(test_data_set)

print('Train input shape:', x_train.shape)
print('Train output shape:', y_train.shape)

if keras_backend.image_data_format() == 'channels_first':
    input_shape = (1, config.IMG_WIDTH, config.IMG_HEIGHT)
else:
    input_shape = (config.IMG_WIDTH, config.IMG_HEIGHT, 1)

model = make_cnn_model(config.IMG_WIDTH, config.IMG_HEIGHT, num_classes, input_shape)
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

save_model(model)

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
