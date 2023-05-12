## Upper Confindence Bound ##


# libraries ---------------------------------------------------------------

library(tidyverse)


# read data ---------------------------------------------------------------

datos <- readr::read_csv("data/Ads_CTR_Optimisation.csv")


# Implementación algoritmo ------------------------------------------------

d <- 10
N <- 10000

number_selections <- integer(d)

sum_rewards <- integer(d)

# sum_rewards[1]/number_selections[1]

ads_selected <- integer(N)

total_rewards <- 0

for(n in 1:N){
  
  max_upper_bound = 0
  ad = 0
  
  for(i in 1:d){
    
    if(number_selections[i] > 0){
    
      average_rewards <- sum_rewards[i]/number_selections[i]
      
      delta_i <- sqrt((3/2) * (log(n)/number_selections[i]))
      
      upper_bound <- average_rewards + delta_i
    
    }else{
      
      upper_bound = 1e400
      
    }
    
    if(upper_bound > max_upper_bound){
      
      max_upper_bound <- upper_bound
      ad <- i
      
    }
    
  }
  
  ads_selected[n] <- ad
  
  number_selections[ad] <- number_selections[ad] + 1
  
  rewards <- datos[n, ad] %>% as.numeric()
  
  sum_rewards[ad] <- sum_rewards[ad] + rewards
  
  total_rewards <- total_rewards + rewards
  
}
  


# visualización de resultados ---------------------------------------------

data.frame(ads_selected) %>% 
  ggplot2::ggplot(aes(ads_selected)) +
  geom_bar(fill = "darkred") +
  labs(title = "Frecuencia de anuncios",
       x = "Anuncio",
       y = "Frecuencia") +
  theme_bw()
