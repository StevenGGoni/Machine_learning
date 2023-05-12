## Random Forest ## 

# Para regresión


# libraries ---------------------------------------------------------------

library(tidyverse)
library(randomForest)

# read data ---------------------------------------------------------------

position_salaries <- readr::read_csv("data/Position_Salaries.csv") %>% 
  janitor::clean_names()

position_salaries <- position_salaries %>% 
  dplyr::select(-position)

# ajuste del modelo -------------------------------------------------------

set.seed(1234)

regresion <- position_salaries %>% 
  randomForest::randomForest(salary ~ ., data = .)

# visualizacion -----------------------------------------------------------

position_salaries %>% 
  dplyr::mutate(ajustados = predict(regresion)) %>% 
  ggplot2::ggplot(aes(level, y = salary)) + 
  geom_point(color = "darkred") +
  geom_line(aes(y = ajustados), color = "darkblue") +
  labs(title = "Random Forest (Regresion)",
       x = "Nivel del empleado",
       y = "Salario ($)") +
  theme_classic()


# predicción --------------------------------------------------------------

predict(regresion, data.frame(level = 6.5))

