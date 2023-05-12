## CART ##

# Uso de CART para regresión, usa la suma de cuadrados


# Libraries ---------------------------------------------------------------

library(tidyverse)
library(rpart)
library(rpart.plot)

# read data ---------------------------------------------------------------

position_salaries <- readr::read_csv("data/Position_Salaries.csv") %>% 
  janitor::clean_names()

position_salaries <- position_salaries %>% 
  dplyr::select(-position)

# ajuste del modelo -------------------------------------------------------

regresion <- position_salaries %>% 
  rpart::rpart(salary ~ ., data = .,
               control = rpart::rpart.control(minsplit = 2))


# visualizacion -----------------------------------------------------------

## Arbol

rpart.plot::prp(regresion)

## Predicción

position_salaries %>% 
  dplyr::mutate(ajustados = predict(regresion)) %>% 
  ggplot2::ggplot(aes(level, y = salary)) + 
  geom_point(color = "darkred") +
  geom_line(aes(y = ajustados), color = "darkblue") +
  labs(title = "CART (Regresion)",
       x = "Nivel del empleado",
       y = "Salario ($)") +
  theme_classic()


# predicción --------------------------------------------------------------

predict(regresion, data.frame(level = 6.5))
