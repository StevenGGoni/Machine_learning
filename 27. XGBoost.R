## XG boosting ##


# libraries ---------------------------------------------------------------

library(tidyverse)
library(caTools)
library(xgboost)

# read data ---------------------------------------------------------------

datos <- readr::read_csv("data/Churn_Modelling.csv")

datos <- datos %>% 
  dplyr::select(-(1:3)) %>% 
  dplyr::mutate(Geography = factor(Geography),
                Gender = factor(Gender))

str(datos)

datos_xg <- model.matrix(Exited ~ ., data = datos)[,-1]

head(datos_xg)

# Entrenamiento y validación ----------------------------------------------

set.seed(123)

split <- caTools::sample.split(datos_xg[,1], SplitRatio = 0.75)

entrenamiento <- datos_xg %>% 
  subset(split == T)

ent_etiqueta <- datos %>% 
  subset(split == T) %>% 
  .$Exited

validacion <- datos_xg %>% 
  subset(split == F)

val_etiqueta <- datos %>% 
  subset(split == F) %>% 
  .$Exited


# ajustar clasificador ----------------------------------------------------

clasificador <- xgboost::xgboost(data = entrenamiento,
                                 nrounds = 200,
                                 label = ent_etiqueta, 
                                 verbose = TRUE, 
                                 objective = "binary:logistic")

y_pred <- as.numeric(predict(clasificador, validacion)>0.5)

# matriz confusion --------------------------------------------------------

table(val_etiqueta, y_pred)

sum(diag(table(val_etiqueta, y_pred)))/sum(table(val_etiqueta, y_pred))

# validación cruzada ------------------------------------------------------

set.seed(123)

folds <- caret::createFolds(datos$Exited)

precisiones <- folds %>% 
  purrr::map_dbl(function(x){
    
    entrenamiento <- datos[-x, ]
    validacion <- datos[x, ]
    
    modelo <- xgboost::xgboost(data = model.matrix(Exited ~ ., 
                                                   data = entrenamiento)[,-1],
                               nrounds = 200,
                               label = entrenamiento$Exited, 
                               verbose = FALSE, 
                               objective = "binary:logistic")
    
    prediccion <- as.numeric(
      predict(clasificador, 
              model.matrix(Exited ~ ., 
                           data = validacion)[,-1])>0.5)
    
    cm <- table(validacion$Exited, prediccion)
    
    sum(diag(cm))/sum(cm)
    
  })


mean(precisiones); sd(precisiones)

