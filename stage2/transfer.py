import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Sequential
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
import pickle

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

path = r'../Data/'
path_train = os.path.join(path, 'train')
path_train_cat = os.path.join(path_train, 'cats')
path_train_dog = os.path.join(path_train, 'dogs')

path_valid = os.path.join(path, 'valid')
path_valid_cat = os.path.join(path_valid, 'cats')
path_valid_dog = os.path.join(path_valid, 'dogs')

size_train = len(os.listdir(path_train_cat) + os.listdir(path_train_dog))
size_valid = len(os.listdir(path_valid_cat) + os.listdir(path_valid_dog))

# Hyper parameters
batch_size = 64
epochs = 5
img_height = img_width = 150
learning_rate = 1e-3

# Stage1
data_generator = ImageDataGenerator(preprocessing_function=preprocess_input, )

train_data_gen = data_generator.flow_from_directory(
    batch_size=batch_size,
    target_size=(img_height, img_width),
    class_mode='categorical',
    directory=path_train,
)

val_data_gen = data_generator.flow_from_directory(
    batch_size=batch_size,
    target_size=(img_height, img_width),
    class_mode='categorical',
    directory=path_valid,
)

test_data_gen = data_generator.flow_from_directory(
    directory=path,
    classes=['test'],
    target_size=(img_height, img_width),
    shuffle=False,
)

# Stage 2

model = Sequential()
model.add(VGG16(
    include_top=False,
    pooling='avg',
    weights='imagenet',
))

model.add(Dense(2, activation='softmax'))

model.layers[0].trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    train_data_gen,
    epochs=epochs,
    validation_data=val_data_gen,
    steps_per_epoch=size_train // batch_size,
    validation_steps=size_valid // batch_size,
    verbose=1,
)

model.save('../SavedModels/stage_two_model.h5')

with open("../SavedHistory/stage_two_history", "wb") as file:
    pickle.dump(history.history, file, protocol=pickle.HIGHEST_PROTOCOL)