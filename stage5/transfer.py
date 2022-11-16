import tensorflow as tf
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Sequential
from keras.layers import Dense, Dropout, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
import numpy as np
import os
import pickle
from keras.models import load_model

if __name__ == '__main__':
    # Write your code below
    model = load_model('../SavedModels/stage_four_model.h5')

    # Fine tuning

    # Hyper parameters
    batch_size = 16
    epochs = 10
    img_height = img_width = 224
    learning_rate = 1e-5

    # call_back = EarlyStopping(monitor='loss', patience=3)

    model.layers[0].trainable = True

    path = r'../Data'
    path_train = os.path.join(path, 'train')
    path_train_cat = os.path.join(path_train, 'cats')
    path_train_dog = os.path.join(path_train, 'dogs')

    path_valid = os.path.join(path, 'valid')
    path_valid_cat = os.path.join(path_valid, 'cats')
    path_valid_dog = os.path.join(path_valid, 'dogs')

    size_train = len(os.listdir(path_train_cat) + os.listdir(path_train_dog))
    size_valid = len(os.listdir(path_valid_cat) + os.listdir(path_valid_dog))

    aug_data_generator = ImageDataGenerator(preprocessing_function=preprocess_input,
                                            rotation_range=30,
                                            horizontal_flip=True,
                                            shear_range=0.2,
                                            zoom_range=0.2,
                                            )

    train_data_gen = aug_data_generator.flow_from_directory(
        batch_size=batch_size,
        target_size=(img_height, img_width),
        class_mode='categorical',
        directory=path_train,
    )

    val_data_gen = aug_data_generator.flow_from_directory(
        batch_size=batch_size,
        target_size=(img_height, img_width),
        class_mode='categorical',
        directory=path_valid,
    )

    test_data_generator = ImageDataGenerator(preprocessing_function=preprocess_input,)
    test_data_gen = test_data_generator.flow_from_directory(
        directory=path,
        classes=['test'],
        target_size=(img_height, img_width),
        shuffle=False,
    )

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

    probabilities = np.argmax(model.predict(test_data_gen), axis=1)

    # print(probabilities)

    with open("../SavedHistory/stage_five_history", "wb") as file:
        pickle.dump(probabilities, file, protocol=pickle.HIGHEST_PROTOCOL)
