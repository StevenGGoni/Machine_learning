# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 12:03:49 2022

@author: Admin
"""

## Redes neuronales convolucionales CNN ##

# =============================================================================
# Libraries
# =============================================================================

import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# =============================================================================
# Inicializar a CNN
# =============================================================================

clasificador = Sequential()

## Convolución

clasificador.add(Conv2D(filters = 32, kernel_size = (3, 3),
                        input_shape = (64, 64, 3),
                        activation = "relu"))

## Max Pooling
clasificador.add(MaxPooling2D(pool_size = 2, strides = 2))

## Flatening

clasificador.add(Flatten())

## Full connection

clasificador.add(Dense(units = 128, 
                       kernel_initializer = "uniform",
                       activation = "relu"))

clasificador.add(Dense(units = 1, # 1 por ser capa final
                       kernel_initializer = "uniform",
                       activation = "sigmoid"))

# =============================================================================
# Ajustar CNN
# =============================================================================

clasificador.compile(optimizer = "adam", loss = "binary_crossentropy",
                     metrics = "accuracy")

## Ajustar la CNN a las imágenes a entrenar

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('data/dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('data/dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

clasificador.fit(x = training_set,
                 validation_data = test_set,
                 epochs = 25)

# =============================================================================
# Hacer predicciones
# =============================================================================

from keras import utils

test_image = utils.load_img('data/dataset/obi_the_chow/obi.jpg', 
                            target_size = (64, 64))

test_image = utils.img_to_array(test_image)

test_image = np.expand_dims(test_image, axis = 0)

result = clasificador.predict(test_image)

training_set.class_indices

if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'

print(prediction)