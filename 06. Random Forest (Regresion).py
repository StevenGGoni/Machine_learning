# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 16:06:32 2021

@author: Admin
"""

## Random Forest

# Usado para regresión

# =============================================================================
# Libraries
# =============================================================================

import pandas as pd
#import numpy as np
import matplotlib.pyplot as plt
import sklearn.ensemble as ens

# =============================================================================
# Read data
# =============================================================================

position_salaries = pd.read_csv("data\Position_Salaries.csv")

x = position_salaries.iloc[:, 1:2].values # es mejor que siempre sea matriz
y = position_salaries.iloc[:, 2].values

# =============================================================================
# Ajustar Random Forest
# =============================================================================

regresion = ens.RandomForestRegressor(n_estimators = 100,
                                      random_state = 0)
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
