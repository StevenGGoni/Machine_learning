# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 19:17:00 2021

@author: Admin
"""

## Naïve Bayes ##

# =============================================================================
# Libraries
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing as sklp
import sklearn.model_selection as sklm
import sklearn.naive_bayes as skln
import sklearn.metrics as sklmetric

# =============================================================================
# Read data
# =============================================================================

social = pd.read_csv("data\Social_Network_Ads.csv")

x = social.iloc[:, [2,3]].values
y = social.iloc[:, 4].values

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
# Ajustar modelo de regresion
# =============================================================================

clasificador = skln.GaussianNB()
clasificador.fit(x_train, y_train)

# =============================================================================
# Predicción de resultados 
# =============================================================================

y_pred = clasificador.predict(x_test)

# =============================================================================
# Matriz de confusion
# =============================================================================

cm = sklmetric.confusion_matrix(y_test, y_pred)

# =============================================================================
# Representación gráfica
# =============================================================================

from matplotlib.colors import ListedColormap

x_set, y_set = x_test, y_test
X1, X2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, clasificador.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Clasificador (Conjunto de Test)')
plt.xlabel('Edad')
plt.ylabel('Sueldo Estimado')
plt.legend()
plt.show()