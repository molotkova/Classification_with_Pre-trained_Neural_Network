import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import keras
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Sequential
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import requests
from zipfile import ZipFile

if __name__ == '__main__':

    if not os.path.exists('../Data'):
        os.mkdir('../Data')

    # if not os.path.exists('../ImageData'):
    #     os.mkdir('../ImageData')

    if not os.path.exists('../SavedModels'):
        os.mkdir('../SavedModels')

    if not os.path.exists('../SavedHistory'):
        os.mkdir('../SavedHistory')

    # Download data if it is unavailable.
    if 'cats-and-dogs-images.zip' not in os.listdir('../Data'):
        print('Image dataset loading.')
        url = "https://www.dropbox.com/s/jgv5zpw41ydtfww/cats-and-dogs-images.zip?dl=1"
        r = requests.get(url, allow_redirects=True)
        open('../Data/cats-and-dogs-images.zip', 'wb').write(r.content)
        print('Loaded.')

        print("\nExtracting files")
        with ZipFile('../Data/cats-and-dogs-images.zip', 'r') as zip:
            zip.extractall(path="../Data")
            print("Completed.")

    # Type your code here

    batch_size = 64
    epochs = 5
    img_height = img_width = 150
    size_train = None
    size_valid = None

    data_generator = ImageDataGenerator(preprocessing_function=preprocess_input,)

    train_data_gen = data_generator.flow_from_directory(
        batch_size=batch_size,
        target_size=(img_height, img_width),
        class_mode='categorical',
        directory=r'../Data/train',
    )

    val_data_gen = data_generator.flow_from_directory(
        batch_size=batch_size,
        target_size=(img_height, img_width),
        class_mode='categorical',
        directory=r'../Data/valid',
    )

    test_data_gen = data_generator.flow_from_directory(
        directory=r'../Data/',
        classes=['test'],
        target_size=(img_height, img_width),
        shuffle=False,
    )

    print(img_height, img_width, batch_size, test_data_gen.shuffle)