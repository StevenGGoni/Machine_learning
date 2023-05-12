## Regresión lineal simple ##

# libraries ---------------------------------------------------------------

library(tidyverse)
library(caTools)
library(broom)

# read data ---------------------------------------------------------------

salarios <- readr::read_delim("data/Salary_Data.csv") # usé read_delim para 
                        # probar el "Delimiter guessing" de readr 2.0.0
                        # caso contrario, read_csv() basta y sobra

# entrenamiento y validacion ----------------------------------------------

set.seed(123)

split <- caTools::sample.split(salarios$Salary, SplitRatio = 2/3)

entrenamiento <- salarios %>% 
  subset(split == T)

validacion <- salarios %>% 
  subset(split == F)


# regresion lineal simple -------------------------------------------------

regresion <- entrenamiento %>% 
  lm(Salary ~ ., data = .)

summary(regresion)

# Visualización de resultados ---------------------------------------------

regresion %>% 
  broom::augment() %>% # para probar el tidy
  ggplot2::ggplot(aes(x = YearsExperience, y = Salary)) +
  geom_point(color = "darkred") + 
  geom_line(aes(y = .fitted), color = "darkblue") +
  labs(title = "Sueldo vs Años de experiencia \n(Entrenamiento)",
       y = "Salario",
       x = "Años de experiencia") +
  theme_classic()

ggplot2::ggplot() +
  geom_point(aes(x = validacion$YearsExperience, 
                 y = validacion$Salary),
             color = "darkred") + 
  geom_line(aes(x = entrenamiento$YearsExperience,
                y = regresion$fitted.values), 
            color = "darkblue") +
  labs(title = "Sueldo vs Años de experiencia \n(Validacion)",
       y = "Salario",
       x = "Años de experiencia") +
  theme_classic()
