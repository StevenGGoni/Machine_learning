## Regresión polinomial ##

# libraries ---------------------------------------------------------------

library(tidyverse)


# read data ---------------------------------------------------------------

position_salaries <- readr::read_csv("data/Position_Salaries.csv")

position_salaries <- position_salaries %>% 
  dplyr::select(-Position) %>% 
  dplyr::mutate(Level2 = Level^2,
                Level3 = Level^3,
                Level4 = Level^4)

# regresion ---------------------------------------------------------------

regresion <- position_salaries %>% 
  lm(Salary ~ ., data = .)

summary(regresion)

# Visualización -----------------------------------------------------------
regresion %>% 
  broom::augment() %>% 
  ggplot2::ggplot(aes(Level, y = Salary)) + 
  geom_point(color = "darkred") +
  geom_line(aes(y = .fitted), color = "darkblue") +
  labs(title = "Regresión polinomial",
       x = "Nivel del empleado",
       y = "Salario ($)") +
  theme_classic()


# Prediccion --------------------------------------------------------------

predict(regresion, data.frame(Level = 6.5,
                              Level2 = 6.5^2,
                              Level3 = 6.5^3,
                              Level4 = 6.5^4))
