# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 16:47:04 2021

@author: Admin
"""

# Regresion polinómica

# =============================================================================
# Libraries
# =============================================================================

import pandas as pd
import matplotlib.pyplot as plt
import sklearn.preprocessing as sklp
import sklearn.linear_model as skllm


# =============================================================================
# Read data 
# =============================================================================

position_salaries = pd.read_csv("data\Position_Salaries.csv")

x = position_salaries.iloc[:, 1:2].values # es mejor que siempre sea matriz
y = position_salaries.iloc[:, 2].values

# =============================================================================
# Ajustar modelo de regresion 
# =============================================================================

# Es necesario transformar los datos originales en una matriz de sus polinomios
polinomial = sklp.PolynomialFeatures(degree = 4)
x_polinomial = polinomial.fit_transform(x)

regresion = skllm.LinearRegression()
regresion.fit(x_polinomial, y)

# =============================================================================
# Visualizacion
# =============================================================================

plt.scatter(x, y, color = "darkred")
plt.plot(x, regresion.predict(x_polinomial), # ojo: usar polinomial
         color = "darkblue")
plt.title("Modelo de regresión polinomial")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo en $")
plt.show()

# =============================================================================
# Prediccion
# =============================================================================

regresion.predict(polinomial.fit_transform([[6.5]]))
