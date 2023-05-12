# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 21:00:08 2021

@author: Admin
"""

### Algoritmo de regresión lineal simple en py ###

# =============================================================================
# libraries
# =============================================================================

import pandas as pd
#import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection as sklm
import sklearn.linear_model as skllm

# =============================================================================
# read data
# =============================================================================

salarios = pd.read_csv("data\Salary_data.csv")

# =============================================================================
# Datos entrenamiento y validacion
# =============================================================================

x_salarios = salarios.iloc[:, :-1].values
y_salarios = salarios.iloc[:, 1].values

x_train, x_test, y_train, y_test = sklm.train_test_split(x_salarios, 
                                                         y_salarios, 
                                                         test_size = 1/3,
                                                         random_state = 0)

# =============================================================================
# Modelo regresión lineal simple
# =============================================================================

regresion = skllm.LinearRegression()
regresion.fit(x_train, y_train)

y_pred = regresion.predict(x_test)

# =============================================================================
# Visualización
# =============================================================================

plt.scatter(x_train, y_train, color = "darkred")
plt.plot(x_train, regresion.predict(x_train), color = "darkblue")
plt.title("Sueldo vs Años de experiencia \nEntrenamiento")
plt.xlabel("Años de experiencia")
plt.ylabel("Sueldo ($)")
plt.show()

# con datos entrenamiento
plt.scatter(x_test, y_test, color = "darkred")
plt.plot(x_train, regresion.predict(x_train), color = "darkblue")
plt.title("Sueldo vs Años de experiencia \nValidación")
plt.xlabel("Años de experiencia")
plt.ylabel("Sueldo ($)")
plt.show()