# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 10:25:26 2022

@author: Admin
"""

## Upper Confidence Bound ##

# =============================================================================
# libraries
# =============================================================================

import pandas as pd
import matplotlib.pyplot as plt
import math


# =============================================================================
# read data
# =============================================================================

data = pd.read_csv("data/Ads_CTR_Optimisation.csv")


# =============================================================================
# Algoritmo de UCB
# =============================================================================

# Como no existe paquetería que implemente este algoritmo, lo vamos a programar
# manualmente.

N = 10000
d = 10
number_of_selections = [0] * d
sums_of_rewards = [0] * d
ads_selected = []
total_reward = 0

for n in range(0, N):
    
    max_upper_bound = 0
    ad = 0
    
    for i in range(0, d):
        
        if(number_of_selections[i]>0):
            
            average_reward = sums_of_rewards[i] / number_of_selections[i]
            delta_i = math.sqrt(3/2*math.log(n+1)/number_of_selections[i])
            upper_bound = average_reward + delta_i
            
        else:
            
            upper_bound = 1e400
            
        if upper_bound > max_upper_bound:
            
            max_upper_bound = upper_bound
            ad = i
            
    ads_selected.append(ad)
    number_of_selections[ad] = number_of_selections[ad] + 1
    reward = data.values[n, ad]
    sums_of_rewards[ad] = sums_of_rewards[ad] + reward
    total_reward = total_reward + reward
    
# =============================================================================
# Visualización de resultados
# =============================================================================

plt.hist(ads_selected)
plt.title("Histograma de anuncios")
plt.xlabel("ID del anuncio")
plt.ylabel("Frecuencia de visualización")
plt.show()