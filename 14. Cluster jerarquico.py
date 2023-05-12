# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 14:05:08 2021

@author: Admin
"""

## Jerarquico ##

# =============================================================================
# libraries
# =============================================================================

import pandas as pd
import matplotlib.pyplot as plt
# import sklearn.preprocessing as sklp
import sklearn.cluster as sklc
import scipy.cluster.hierarchy as sch

# =============================================================================
# read data
# =============================================================================

mall = pd.read_csv("data\Mall_Customers.csv")

x = mall.iloc[:, [3, 4]].values #ingresos y puntuacion

# =============================================================================
# Dendrograma
# =============================================================================

dendrogram = sch.dendrogram(sch.linkage(x, method = "ward"))
plt.title("Dendrograma")
plt.xlabel("Clientes")
plt.ylabel("Distancia Euclidea")
plt.show()

# =============================================================================
# Ajustar modelo de cluster
# =============================================================================

cluster = sklc.AgglomerativeClustering(n_clusters = 5, affinity = "euclidean",
                                       linkage = "ward")

y_cluster = cluster.fit_predict(x)

# =============================================================================
# Visualización de los cluster
# =============================================================================

plt.scatter(x[y_cluster == 0, 0], x[y_cluster == 0, 1], 
            c = "darkred", s = 100, label = "Cluster 1")
plt.scatter(x[y_cluster == 1, 0], x[y_cluster == 1, 1], 
            c = "darkblue", s = 100, label = "Cluster 2")
plt.scatter(x[y_cluster == 2, 0], x[y_cluster == 2, 1], 
            c = "green", s = 100, label = "Cluster 3")
plt.scatter(x[y_cluster == 3, 0], x[y_cluster == 3, 1], 
            c = "cyan", s = 100, label = "Cluster 4")
plt.scatter(x[y_cluster == 4, 0], x[y_cluster == 4, 1], 
            c = "magenta", s = 100, label = "Cluster 5")
# plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
#             c = "black", s = 300, label = "Centroides")
plt.title("Cluster de clientres")
plt.xlabel("Ingresos anuales (miles $)")
plt.ylabel("Puntuación de gastos")
plt.legend()
plt.show()
