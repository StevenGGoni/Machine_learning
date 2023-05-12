## Support Vector Regression ##

# Básicamente tiene la misma estructura que SVM, pero que empleado para predecir
# en lugar de clasificar

# libraries ---------------------------------------------------------------

library(tidyverse)
library(e1071)


# read data ---------------------------------------------------------------

position_salaries <- readr::read_csv("data/Position_Salaries.csv") %>% 
  janitor::clean_names()

position_salaries <- position_salaries %>% 
  dplyr::select(-position)

# ajuste del modelo -------------------------------------------------------

regresion <- position_salaries %>% 
  e1071::svm(salary ~ ., 
             type = "eps-regression", # especificar que es una regresión
             kernel = "radial", # el problema no es lineal
             data = .)

# Visualización -----------------------------------------------------------

## Los valores atípicos suelen ser penalizados por las SVM, por eso el
## nivel 10 no se predice "tan bien" como deberia

position_salaries %>% 
  dplyr::mutate(ajustados = regresion$fitted) %>% 
  ggplot2::ggplot(aes(level, y = salary)) + 
  geom_point(color = "darkred") +
  geom_line(aes(y = ajustados), color = "darkblue") +
  labs(title = "SVR",
       x = "Nivel del empleado",
       y = "Salario ($)") +
  theme_classic()

# prediccion --------------------------------------------------------------

predict(regresion, data.frame(level = 6.5))
