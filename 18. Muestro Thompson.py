# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 20:30:54 2022

@author: Admin
"""

## Muestreo de Thompson ##

# =============================================================================
# libraries
# =============================================================================

import pandas as pd
import matplotlib.pyplot as plt
import random


# =============================================================================
# read data
# =============================================================================

data = pd.read_csv("data/Ads_CTR_Optimisation.csv")


# =============================================================================
# Algoritmo Thompson
# =============================================================================


N = 10000
d = 10

number_rewards_1 = [0] * d
number_rewards_0 = [0] * d

ads_selected = []
total_reward = 0


for n in range(0, N):
    
    max_random = 0
    
    ad = 0
    
    for i in range(0, d):
        
        random_beta = random.betavariate(number_rewards_1[i] + 1, 
                                         number_rewards_0[i] + 1)
        
        if random_beta > max_random:
            
            max_random = random_beta
            
            ad = i
            
    ads_selected.append(ad)
    reward = data.values[n, ad]
        
    if reward == 1:
            
        number_rewards_1[ad] = number_rewards_1[ad] + 1
            
    else:
            
        number_rewards_0[ad] = number_rewards_0[ad] + 1
        
    total_reward = total_reward + reward


# =============================================================================
# Visualización de resultados
# =============================================================================

plt.hist(ads_selected)
plt.title("Histograma de anuncios")
plt.xlabel("ID del anuncio")
plt.ylabel("Frecuencia de visualización")
plt.show()