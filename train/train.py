import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import keras

from keras.applications import ResNet50
from keras.models import Sequential
from keras.layers import Dense, Flatten, GlobalAveragePooling2D
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

num_classes = 5
#resnet_weights_path = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

my_new_model = Sequential()
my_new_model.add(ResNet50(weights= None, include_top=False, pooling='avg'))
my_new_model.add(Dense(num_classes, activation='softmax'))

# Say not to train first layer (ResNet) model. It is already trained
my_new_model.layers[0].trainable = False

my_new_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])


from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator

image_size = 224
data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = data_generator.flow_from_directory(
        '/Users/raza/Documents/programs/research/Research-transfer-learning/test_data/Train',
        target_size=(image_size, image_size),
        batch_size=50,
        class_mode='categorical')

validation_generator = data_generator.flow_from_directory(
        '/Users/raza/Documents/programs/research/Research-transfer-learning/test_data/Val',
        target_size=(image_size, image_size),
        class_mode='categorical')


history = my_new_model.fit_generator(
          train_generator,
          steps_per_epoch=70,
          epochs=10,
          validation_data=validation_generator,
          validation_steps=1)