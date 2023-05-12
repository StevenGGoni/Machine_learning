## NLP ##


# libraries ---------------------------------------------------------------

library(tidyverse)
library(tm)
library(SnowballC)
library(caTools)
library(e1071)

# read data ---------------------------------------------------------------

datos <- readr::read_tsv("data/Restaurant_Reviews.tsv")


# limpiar texto -----------------------------------------------------------

corpus <- tm::VectorSource(datos$Review) %>% 
  tm::VCorpus() %>% 
  tm::tm_map(tm::content_transformer(stringr::str_to_lower)) %>% 
  tm::tm_map(tm::removeNumbers) %>% 
  tm::tm_map(tm::removePunctuation) %>% 
  tm::tm_map(tm::removeWords, tm::stopwords(kind = "en")) %>% 
  tm::tm_map(tm::stemDocument) %>% 
  tm::tm_map(tm::stripWhitespace)


# matriz dispersa (bag of words) ------------------------------------------

dtm <- tm::DocumentTermMatrix(corpus) %>% 
  tm::removeSparseTerms(0.999) %>% 
  as.matrix() %>% 
  as.data.frame() %>% 
  dplyr::mutate(liked = as.factor(datos$Liked))


# algritmo de clasificación -----------------------------------------------

## Random Forest

# Entrenamiento y validación ----------------------------------------------

set.seed(123)

split <- caTools::sample.split(dtm$liked, SplitRatio = 0.80)

entrenamiento <- dtm %>% 
  subset(split == T)

validacion <- dtm %>% 
  subset(split == F)


# ajustar clasificacion ---------------------------------------------------

clasificador <- randomForest::randomForest(x = entrenamiento[, -692],
                                           y = entrenamiento$liked)


# predicción de resultados ------------------------------------------------

y_pred <- predict(clasificador, validacion, type = "class")

# matriz confusion --------------------------------------------------------

table(validacion$liked, y_pred)
