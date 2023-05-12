# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 15:00:03 2022

@author: Admin
"""

## xgboost ##

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
# División de entrenamiento y test
# =============================================================================

x_train, x_test, y_train, y_test = sklm.train_test_split(x, y, test_size= 0.25,
                                                         random_state = 0)

# =============================================================================
# Ajustar XGBoost
# =============================================================================

from xgboost import XGBClassifier as XGB

clasificador = XGB()
clasificador.fit(x_train, y_train)

# =============================================================================
# Predicción de resultados 
# =============================================================================

y_pred = clasificador.predict(x_test)

# =============================================================================
# Matriz de confusion
# =============================================================================

cm = sklmetric.confusion_matrix(y_test, y_pred)
cm

np.sum(np.diag(cm))/np.sum(cm)

# =============================================================================
# Validacion cruzada
# =============================================================================

from sklearn.model_selection import cross_val_score as CV

precisiones =  CV(estimator = clasificador, X = x_train, y = y_train, cv = 10)

precisiones.mean()
precisiones.std()
