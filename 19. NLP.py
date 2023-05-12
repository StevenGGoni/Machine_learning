# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 10:26:17 2022

@author: Admin
"""

## NLP ##

# =============================================================================
# libraries
# =============================================================================

#import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import sklearn.model_selection as sklm
import sklearn.naive_bayes as skln
import sklearn.metrics as sklmetric
import re # regular expressions
import nltk

# =============================================================================
# read data
# =============================================================================

data = pd.read_csv("data\Restaurant_Reviews.tsv", delimiter = "\t", 
                   quoting = 3) # ignorar comillas dobles

# =============================================================================
# limpiar el texto
# =============================================================================

# La idea es quedarse solo con las palabras que son relevantes para decidir
# en sentido negativo o positivo. 

nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []

for i in range(0, 1000):

    review = re.sub("[^a-zA-Z]", " ", data["Review"][i])
    review = review.lower()
    review = review.split() #separada por espacios
    
    ps = PorterStemmer()
    
    review = [ps.stem(word) for word in review if not word in set(stopwords.words("english"))]
    review = " ".join(review)
    
    corpus.append(review)

# =============================================================================
# Crear el Bag of Words
# =============================================================================

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = data.iloc[:, 1].values

# =============================================================================
# Aplicación de un algoritmo de clasificación
# =============================================================================

# NAIVE BAYES

# =============================================================================
# División de entrenamiento y test
# =============================================================================

x_train, x_test, y_train, y_test = sklm.train_test_split(X, y, test_size= 0.20,
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

