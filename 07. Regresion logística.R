## Regresion logística ##


# libraries ---------------------------------------------------------------

library(tidyverse)
library(caTools)


# read data ---------------------------------------------------------------

social <- readr::read_csv("data/Social_Network_Ads.csv", 
                          col_select = -c(1,2)) %>% 
  janitor::clean_names()


# escalado de variables ---------------------------------------------------

social <- social %>% 
  dplyr::mutate(age = scale(age) %>% as.numeric(),
                estimated_salary = scale(estimated_salary) %>% as.numeric())

# Entrenamiento y validación ----------------------------------------------

set.seed(123)

split <- caTools::sample.split(social$purchased, SplitRatio = 0.75)

entrenamiento <- social %>% 
  subset(split == T)

validacion <- social %>% 
  subset(split == F)


# ajustar clasificacion ---------------------------------------------------

clasificador <- entrenamiento %>% 
  glm(purchased ~ ., data = ., family = "binomial")


# predicción de resultados ------------------------------------------------

y_pred <- dplyr::if_else(predict(clasificador, 
                                 validacion, type = "response") > 0.5, 
                         1, 0)

# matriz confusion --------------------------------------------------------

table(validacion$purchased, y_pred)


# Visualizacion -----------------------------------------------------------

library(RColorBrewer)

age <- seq(min(validacion$age) - 1,
           max(validacion$age) + 1, by = 0.01)

estimated_salary <- seq(min(validacion$estimated_salary) - 1,
                        max(validacion$estimated_salary) + 1, by = 0.01)

tidyr::expand_grid(age, estimated_salary) %>% 
  predict(clasificador, newdata = . , type = "response") %>% 
  tibble::tibble(tidyr::expand_grid(age, estimated_salary), purchased = .) %>% 
  dplyr::mutate(purchased = if_else(purchased > 0.5, 1, 0)) %>% 
  ggplot2::ggplot(aes(age, estimated_salary)) + 
  geom_tile(aes(fill = factor(purchased)), alpha = 0.05) +
  geom_point(data = validacion, 
             aes(age, estimated_salary, color = factor(purchased))) +
  labs(title = "Clasificador: KNN",
       subtitle = "(Conjunto de test)",
       x = "Edad",
       y = "Salario estimado",
       fill = NULL,
       color = NULL) +
  scale_fill_brewer(palette = "Dark2") +
  scale_color_brewer(palette = "Dark2") +
  theme_classic()
