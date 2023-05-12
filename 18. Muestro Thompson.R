## Muestreo Thomposon ##


# libraries ---------------------------------------------------------------

library(tidyverse)


# read data ---------------------------------------------------------------

datos <- readr::read_csv("data/Ads_CTR_Optimisation.csv")


# Algoritmo de muestreo de Thompson ---------------------------------------

d <- 10
N <- 10000

number_rewards_1 <- integer(d)
number_rewards_0 <- integer(d)

ads_selected <- integer(0)

total_rewards <- 0

for(n in 1:N){
  
  max_random <- 0
  ad <- 0
  
  for(i in 1:d){

    random_beta <- rbeta(n = 1, 
                         shape1 = number_rewards_1[i] + 1, 
                         shape2 = number_rewards_0[i] + 1)
    
    if(random_beta > max_random){
      
      max_random <- random_beta
      ad <- i
      
    }
    
  }
  
  ads_selected = append(ads_selected, ad)
  rewards <- datos[n, ad]
  
  
  if(rewards == 1){
    
    number_rewards_1[ad] <- number_rewards_1[ad] + 1
    
  }else{
    
    number_rewards_0[ad] <- number_rewards_0[ad] + 1
    
  }
  
  total_rewards = total_rewards + rewards
  
}

# visualización de resultados ---------------------------------------------

data.frame(ads_selected) %>% 
  ggplot2::ggplot(aes(ads_selected)) +
  geom_bar(fill = "darkred") +
  labs(title = "Frecuencia de anuncios",
       x = "Anuncio",
       y = "Frecuencia") +
  theme_bw()