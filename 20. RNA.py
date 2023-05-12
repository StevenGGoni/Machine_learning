# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 10:45:24 2022

@author: Admin
"""

## Redes neuronales artificiales ##

# =============================================================================
# Libraries
# =============================================================================

import numpy as np
import pandas as pd
import sklearn.preprocessing as sklp
import sklearn.model_selection as sklm
import sklearn.compose as sklc
import sklearn.metrics as sklmetric

# =============================================================================
# Read data
# =============================================================================

churn = pd.read_csv("data\Churn_Modelling.csv")

x = churn.iloc[:, 3:13].values
y = churn.iloc[:, 13].values


# =============================================================================
# Datos categóricos
# =============================================================================

label_encoder_x1 = sklp.LabelEncoder()
x[:, 1] = label_encoder_x1.fit_transform(x[:, 1])

label_encoder_x2 = sklp.LabelEncoder()
x[:, 2] = label_encoder_x2.fit_transform(x[:, 2])

ct = sklc.ColumnTransformer(
    [("one_hot_encoder", sklp.OneHotEncoder(categories = "auto"), [1])],
    remainder = "passthrough")

x = np.array(ct.fit_transform(x), dtype = np.float64)

x = x[:,1:]

# =============================================================================
# Escalado de variables
# =============================================================================

sc_x = sklp.StandardScaler()
x = sc_x.fit_transform(x)

# =============================================================================
# División de entrenamiento y test
# =============================================================================

x_train, x_test, y_train, y_test = sklm.train_test_split(x, y, test_size= 0.25,
                                                         random_state = 0)

# =============================================================================
# Importar librerías adicionales
# =============================================================================

import keras
from keras.models import Sequential # inicializar RNA
from keras.layers import Dense

# =============================================================================
# Inicializar la red neuronal
# =============================================================================

clasificador = Sequential()

# Añadir capas de entrada y capas ocultas

clasificador.add(Dense(units = 6, kernel_initializer = "uniform",
                       activation = "relu", input_dim = 11))

## Añadir segunda capa oculta
clasificador.add(Dense(units = 6, kernel_initializer = "uniform",
                       activation = "relu"))

## Añadir capa de salida

clasificador.add(Dense(units = 1, kernel_initializer = "uniform",
                       activation = "sigmoid"))

# =============================================================================
# Ajustar RNA
# =============================================================================
  
clasificador.compile(optimizer = "adam", loss = "binary_crossentropy",
                     metrics = "accuracy")


clasificador.fit(x_train, y_train, batch_size = 10, epochs = 100)

# =============================================================================
# Predicción de resultados 
# =============================================================================

y_pred = clasificador.predict(x_test)

y_pred = (y_pred>0.5)

# =============================================================================
# Matriz de confusion
# =============================================================================

cm = sklmetric.confusion_matrix(y_test, y_pred)
cm
