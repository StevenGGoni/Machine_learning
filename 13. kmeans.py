# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 08:53:32 2021

@author: Admin
"""

## Kmeans##

# =============================================================================
# libraries
# =============================================================================

import pandas as pd
import matplotlib.pyplot as plt
# import sklearn.preprocessing as sklp
import sklearn.cluster as sklc

# =============================================================================
# read data
# =============================================================================

mall = pd.read_csv("data\Mall_Customers.csv")

x = mall.iloc[:, [3, 4]].values #ingresos y puntuacion

# =============================================================================
# método del codo
# =============================================================================

wcss = []

for i in range(1,11):
    
    kmeans = sklc.KMeans(n_clusters = i, init = "k-means++", 
                         max_iter = 300, n_init = 10, random_state = 0)
    
    kmeans.fit(x)
    
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11), wcss)
plt.title("Método del codo")
plt.xlabel("Número de cluster")
plt.ylabel("WCSS")    
plt.show()
    
# =============================================================================
# Ajustar modelo de cluster
# =============================================================================

kmeans = sklc.KMeans(n_clusters = 5, init = "k-means++", 
                         max_iter = 300, n_init = 10, random_state = 0)

y_cluster = kmeans.fit_predict(x)

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
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            c = "black", s = 300, label = "Centroides")
plt.title("Cluster de clientres")
plt.xlabel("Ingresos anuales (miles $)")
plt.ylabel("Puntuación de gastos")
plt.legend()
plt.show()