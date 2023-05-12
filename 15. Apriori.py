# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 14:58:05 2021

@author: Admin
"""

## Apriori ##

# =============================================================================
# libraries
# =============================================================================

import pandas as pd

# =============================================================================
# read data
# =============================================================================

data = pd.read_csv("data\Market_Basket_Optimisation.csv", header=None)

transactions = []

for i in range(0, 7501): 
    transactions.append([str(data.values[i, j]) for j in range(0, 20)])
    
# =============================================================================
# entrenar algoritmo
# =============================================================================

from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2,
                min_length = 2, min_lift = 3)


# =============================================================================
# Visualización resultados 
# =============================================================================

results = list(rules)

results[0]
