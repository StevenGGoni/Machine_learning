## Grid search ##

# libraries ---------------------------------------------------------------

library(tidyverse)
library(caTools)
library(e1071)
library(caret)


# read data ---------------------------------------------------------------

social <- readr::read_csv("data/Social_Network_Ads.csv", 
                          col_select = -c(1,2)) %>% 
  janitor::clean_names()


# escalado de variables ---------------------------------------------------

social <- social %>% 
  dplyr::mutate(age = scale(age) %>% 
                  as.numeric(),
                estimated_salary = scale(estimated_salary) %>% 
                  as.numeric(), 
                purchased = factor(purchased))

# Entrenamiento y validación ----------------------------------------------

set.seed(123)

split <- caTools::sample.split(social$purchased, SplitRatio = 0.75)

entrenamiento <- social %>% 
  subset(split == T)

validacion <- social %>% 
  subset(split == F)


# grid search -------------------------------------------------------------

gs <- caret::train(purchased ~ ., 
                   data = entrenamiento,
                   method = "svmRadial")

gs$results
gs$bestTune

# ajustar clasificacion ---------------------------------------------------

clasificador <- entrenamiento %>% 
  e1071::svm(purchased ~ .,
             type = "C-classification",
             kernel = "radial", # el problema no es lineal
             data = .)

# predicción de resultados ------------------------------------------------

y_pred <- predict(clasificador, validacion)

# matriz confusion --------------------------------------------------------

table(validacion$purchased, y_pred)

# Visualizacion -----------------------------------------------------------

library(RColorBrewer)

age <- seq(min(validacion$age) - 1,
           max(validacion$age) + 1, by = 0.01)

estimated_salary <- seq(min(validacion$estimated_salary) - 1,
                        max(validacion$estimated_salary) + 1, by = 0.01)

tidyr::expand_grid(age, estimated_salary) %>% 
  predict(clasificador, newdata = .) %>% 
  tibble::tibble(tidyr::expand_grid(age, estimated_salary), purchased = .) %>% 
  ggplot2::ggplot(aes(age, estimated_salary)) + 
  geom_tile(aes(fill = factor(purchased)), alpha = 0.05) +
  geom_point(data = validacion, 
             aes(age, estimated_salary, color = factor(purchased))) +
  labs(title = "Clasificador: SVM",
       subtitle = "(Conjunto de test)",
       x = "Edad",
       y = "Salario estimado",
       fill = NULL,
       color = NULL) +
  scale_fill_brewer(palette = "Dark2") +
  scale_color_brewer(palette = "Dark2") +
  theme_classic()