# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 15:58:06 2022

@author: Admin
"""

## LDA ##

# Regresion logística

# =============================================================================
# Libraries
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing as sklp
import sklearn.model_selection as sklm
import sklearn.linear_model as skll
import sklearn.metrics as sklmetric

# =============================================================================
# Read data
# =============================================================================

vinos = pd.read_csv("data\Wine.csv")

x = vinos.iloc[:, 0:13].values
y = vinos.iloc[:, 13].values

# =============================================================================
# Escalado de variables
# =============================================================================

sc_x = sklp.StandardScaler()
x = sc_x.fit_transform(x)

# =============================================================================
# Reducir la dimensión con ACP
# =============================================================================

from sklearn.decomposition import PCA

pca = PCA(n_components = 2)

x = pca.fit_transform(x)

pca.explained_variance_ratio_

# =============================================================================
# División de entrenamiento y test
# =============================================================================

x_train, x_test, y_train, y_test = sklm.train_test_split(x, y, test_size= 0.25,
                                                         random_state = 0)

# =============================================================================
# Ajustar modelo de regresion
# =============================================================================

clasificador = skll.LogisticRegression(random_state = 0)

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
# =============================================================================
# Representación gráfica
# =============================================================================

from matplotlib.colors import ListedColormap

x_set, y_set = x_test, y_test
X1, X2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1,
                               stop = x_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = x_set[:, 1].min() - 1, 
                               stop = x_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, clasificador.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('Clasificador (Conjunto de Test)')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.legend()
plt.show()