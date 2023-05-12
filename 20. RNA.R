## Redes neuronales artificiales ##


# libraries ---------------------------------------------------------------

library(tidyverse)
library(caTools)
library(h2o)

# read data ---------------------------------------------------------------

datos <- readr::read_csv("data/Churn_Modelling.csv")

datos <- datos %>% 
  dplyr::select(-(1:3)) %>% 
  dplyr::mutate(Geography = factor(Geography, 
                                   levels = c("France", 
                                              "Spain", "Germany"),
                                   labels = c(1, 2, 3)),
                Gender = factor(Gender, 
                                levels = c("Female", "Male"),
                                labels = c(0, 1)),
                Exited = factor(Exited))

str(datos)

# escalado de variables ---------------------------------------------------
# No es necesario escalar ############
# datos <- datos %>% 
#   dplyr::mutate(across(-11, scale)) %>% 
#   purrr::map_df(function(x) {attributes(x) <- NULL; x}) %>% 
#   purrr::modify_at(11, factor)

# Entrenamiento y validación ----------------------------------------------

set.seed(123)

split <- caTools::sample.split(datos$Exited, SplitRatio = 0.75)

entrenamiento <- datos %>% 
  subset(split == T)

validacion <- datos %>% 
  subset(split == F)


# ajustar clasificador ----------------------------------------------------

h2o.init(nthreads = 3)

h2o_train <- as.h2o(as.data.frame(entrenamiento))
h20_test <- as.h2o(as.data.frame(validacion))

clasificador <- h2o.deeplearning(y = "Exited",
                                 training_frame = h2o_train,
                                 activation = "Rectifier",
                                 hidden = c(6, 6),
                                 epochs = 100,
                                 train_samples_per_iteration = -2,
                                 standardize = TRUE) # no es necesario escalar

# predicción de resultados ------------------------------------------------

pred <- h2o.predict(clasificador, h20_test) %>% 
  as.data.frame()

y_pred <- pred$predict

# matriz confusion --------------------------------------------------------

table(validacion$Exited, y_pred)

sum(diag(table(validacion$Exited, y_pred)))/sum(table(validacion$Exited, y_pred))

# cerrar la sesión de h2o -------------------------------------------------

h2o.shutdown(prompt = FALSE)

