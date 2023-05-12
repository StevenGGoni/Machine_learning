## Regresión lineal múltiple ##

# libraries ---------------------------------------------------------------

library(tidyverse)
library(caTools)

# read data ---------------------------------------------------------------

startup <- readr::read_csv("data/Startups.csv") %>% 
  janitor::clean_names()


# Entrenamiento y validación ----------------------------------------------

set.seed(123)

split <- caTools::sample.split(startup$profit, SplitRatio = 0.8)

entrenamiento <- startup %>% 
  subset(split == T)

validacion <- startup %>% 
  subset(split == F)

# Regresión ---------------------------------------------------------------

regresion <- lm(profit ~ ., data = entrenamiento)

summary(regresion)

predict(regresion, validacion)

# Eliminación hacia atrás -------------------------------------------------
### Con 0.05 de alfa
## Manual

manual <- entrenamiento %>% 
  lm(profit ~ r_d_spend + administration + marketing_spend, 
     data = .) # Sin state
summary(manual)

manual <- entrenamiento %>% 
  lm(profit ~ r_d_spend + marketing_spend, 
     data = .) # Sin administracion
summary(manual)

manual <- entrenamiento %>% 
  lm(profit ~ r_d_spend, 
     data = .) # Sin marketing_spend
summary(manual)


## Automático
### Con AIC
automatico <- step(regresion, direction = "backward",
                   scope = ~ r_d_spend + administration + 
                     marketing_spend + state)

summary(automatico)
