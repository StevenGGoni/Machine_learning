# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 19:47:04 2021

@author: Admin
"""

# Regresión lineal múltiple 

# =============================================================================
# Libraries
# =============================================================================

import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import sklearn.preprocessing as sklp
import sklearn.model_selection as sklm
import sklearn.compose as sklc
import sklearn.linear_model as skllm
import statsmodels.api as sm

# =============================================================================
# Read data 
# =============================================================================

startup = pd.read_csv("data\Startups.csv")

x = startup.iloc[:, :-1].values
y = startup.iloc[:, 4].values


# =============================================================================
# Datos categoricos: variables dummy
# =============================================================================

label_encoder_x = sklp.LabelEncoder()
x[:, 3] = label_encoder_x.fit_transform(x[:, 3])
onehotencoder = sklc.make_column_transformer((sklp.OneHotEncoder(), [3]), 
                                        remainder = "passthrough")
x = onehotencoder.fit_transform(x)

# Se ordenan alfabéticamente, en este caso: California, Florida, New York
# Hay que eliinar una dummy, solo se necesitan 2 **Colinealidad**
x = x[:, 1:] # quitamos Carolina

# =============================================================================
# Datos entrenamiento y validacion 
# =============================================================================

x_train, x_test, y_train, y_test = sklm.train_test_split(x, 
                                                         y, 
                                                         test_size = 0.2,
                                                         random_state = 0)

# =============================================================================
# Ajustar modelos de regresión
# =============================================================================

regresion = skllm.LinearRegression()
regresion.fit(x_train, y_train)

y_pred = regresion.predict(x_test)

# =============================================================================
# Selección hacia atrás
# =============================================================================

## Agregar el intercepto

x = np.append(arr = np.ones((50, 1)).astype(int), values = x, axis = 1)

# variables independientes estadísticamente significativas
x_opt = x[:, [0, 1, 2, 3, 4, 5]].tolist()
regresion_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regresion_OLS.summary()

## Quitamos ambas dummy (tiene que ser una a una)
x_opt = x[:, [0, 3, 4, 5]].tolist()
regresion_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regresion_OLS.summary()

## Quitamos 4 (Administracion)
x_opt = x[:, [0, 3, 5]].tolist()
regresion_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regresion_OLS.summary()

## Quitamos 5 (Marketing Spend)
x_opt = x[:, [0, 3]].tolist()
regresion_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regresion_OLS.summary()