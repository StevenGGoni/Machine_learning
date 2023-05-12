# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 15:22:38 2021

@author: Admin
"""

# Classification and regression trees

## Usado para regresión, no para clasificación

# =============================================================================
# Libraries
# =============================================================================

import pandas as pd
#import numpy as np
import matplotlib.pyplot as plt
import sklearn.tree as tree

# =============================================================================
# Read data
# =============================================================================

position_salaries = pd.read_csv("data\Position_Salaries.csv")

x = position_salaries.iloc[:, 1:2].values # es mejor que siempre sea matriz
y = position_salaries.iloc[:, 2].values

# =============================================================================
# Ajustar CART
# =============================================================================

regresion = tree.DecisionTreeRegressor(random_state = 0)
regresion.fit(x, y)

# =============================================================================
# Visualizacion
# =============================================================================

plt.scatter(x, y, color = "darkred")
plt.plot(x, regresion.predict(x), color = "darkblue")
plt.title("Modelo CART (Regresion)")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo en $")
plt.show()

# =============================================================================
# Prediccion
# =============================================================================

regresion.predict([[6.5]])
