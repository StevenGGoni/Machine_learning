# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 18:40:28 2021

@author: Admin
"""

#SVR

# =============================================================================
# Libraries
# =============================================================================

import pandas as pd
#import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing as sklp
import sklearn.svm as svm

# =============================================================================
# Read data
# =============================================================================

position_salaries = pd.read_csv("data\Position_Salaries.csv")

x = position_salaries.iloc[:, 1:2].values # es mejor que siempre sea matriz
y = position_salaries.iloc[:, 2].values

# =============================================================================
# Escalado de variables
# =============================================================================

## Escalar las variables es importante para este algoritmo

scl_x = sklp.StandardScaler()
scl_y = sklp.StandardScaler()

x = scl_x.fit_transform(x)
y = scl_y.fit_transform(y.reshape(-1, 1))

# =============================================================================
# Ajustar SVR
# =============================================================================

regresion = svm.SVR(kernel = "rbf") # Kernel gaussiano
regresion.fit(x, y)

# =============================================================================
# Visualizacion
# =============================================================================

## Nótese que se invierte la transformación!

plt.scatter(scl_x.inverse_transform(x), scl_y.inverse_transform(y), 
            color = "darkred")
plt.plot(scl_x.inverse_transform(x), 
         scl_y.inverse_transform(regresion.predict(x)), 
         color = "darkblue")
plt.title("Modelo SVR")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo en $")
plt.show()

# =============================================================================
# Prediccion
# =============================================================================

scl_y.inverse_transform(regresion.predict(scl_x.transform([[6.5]])))
