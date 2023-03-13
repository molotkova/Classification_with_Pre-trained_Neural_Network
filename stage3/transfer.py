import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from keras.models import load_model
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pickle

#if __name__ == '__main__':
#    # Write your code below
#    path_model = r'../SavedModels'
#    path_answer = r'../SavedHistory'
#    path_data = r'../Data'
#
#    # Hyper parameters
#    batch_size = 64
#    epochs = 5
#    img_height = img_width = 150
#    learning_rate = 1e-3
#
#    # Stage3
#    data_generator = ImageDataGenerator(preprocessing_function=preprocess_input,)
#
#    test_data_gen = data_generator.flow_from_directory(
#        directory=path_data,
#        classes=['test'],
#        target_size=(img_height, img_width),
#        shuffle=False,
#    )
#
#    model = load_model(os.path.join(path_model, 'stage_two_model.h5'))
#
#
#    probabilities = np.argmax(model.predict(test_data_gen), axis=1)
#
#    with open("../SavedHistory/stage_three_history", "wb") as file:
#        pickle.dump(probabilities, file, protocol=pickle.HIGHEST_PROTOCOL)


