## K folds ##

# libraries ---------------------------------------------------------------

library(tidyverse)
library(caret)
library(kernlab)


# read data ---------------------------------------------------------------

datos <- readr::read_csv("data/Social_Network_Ads.csv")

# escalado de variables ---------------------------------------------------

datos <- datos %>% 
  dplyr::mutate(across(c(3,4), scale)) %>% 
  dplyr::mutate(across(c(3,4), as.numeric)) %>% 
  dplyr::select(3:5)

# kernel pca --------------------------------------------------------------

k_pca <- kernlab::kpca(~., datos[,-3], kernel = "rbfdot", 
                       features = 2)


datos_k_pca <- data.frame(k_pca@rotated, datos[, 3])



# validación cruzada ------------------------------------------------------

set.seed(123)

folds <- caret::createFolds(datos_k_pca$Purchased)

precisiones <- folds %>% 
  purrr::map_dbl(function(x){
    
    entrenamiento <- datos_k_pca[-x, ]
    validacion <- datos_k_pca[x, ]
    
    modelo <- entrenamiento %>% 
      glm(Purchased ~ ., data = ., family = "binomial")
    
    prediccion <- dplyr::if_else(predict(modelo, 
                                         validacion, 
                                         type = "response") > 0.5, 
                                 1, 0)
    
    cm <- table(validacion$Purchased, prediccion)
    
    sum(diag(cm))/sum(cm)
    
  })


mean(precisiones); sd(precisiones)


# ajustar clasificacion ---------------------------------------------------

clasificador <- entrenamiento %>% 
  glm(Purchased ~ ., data = ., family = "binomial")


# predicción de resultados ------------------------------------------------

y_pred <- dplyr::if_else(predict(clasificador, 
                                 validacion, type = "response") > 0.5, 
                         1, 0)

# matriz confusion --------------------------------------------------------

table(validacion$Purchased, y_pred)


# Visualizacion -----------------------------------------------------------

X1 <- seq(min(validacion$X1) - 1,
          max(validacion$X1) + 1, by = 0.05)

X2 <- seq(min(validacion$X2) - 1,
          max(validacion$X2) + 1, by = 0.05)

tidyr::expand_grid(X1, X2) %>% 
  predict(clasificador, newdata = . , type = "response") %>% 
  tibble::tibble(tidyr::expand_grid(X1, X2), Purchased = .) %>% 
  dplyr::mutate(Purchased = if_else(Purchased > 0.5, 1, 0)) %>% 
  ggplot2::ggplot(aes(X1, X2)) + 
  geom_tile(aes(fill = factor(Purchased)), alpha = 0.25) +
  geom_point(data = validacion, 
             aes(X1, X2, color = factor(Purchased))) +
  labs(title = "Clasificador: Logístico",
       subtitle = "(Conjunto de test)",
       x = "X1",
       y = "X2",
       fill = NULL,
       color = NULL) +
  scale_fill_brewer(palette = "Dark2") +
  scale_color_brewer(palette = "Dark2") +
  theme_classic()
