## PCA ##


# libraries ---------------------------------------------------------------

library(tidyverse)
library(FactoMineR)


# read data ---------------------------------------------------------------

datos <- readr::read_csv("data/Wine.csv")



# pca ---------------------------------------------------------------------

pca <- FactoMineR::PCA(datos[,-14], scale.unit = TRUE, ncp = 2)

pca$eig

datos_pca <- data.frame(pca$ind$coord, datos[, 14])


# pca con caret -----------------------------------------------------------

pca_caret <- caret::preProcess(datos[,-14], pcaComp = 2, method = "pca")

datos_caret <- predict(pca_caret, datos)

head(datos_caret)

# Entrenamiento y validación ----------------------------------------------

set.seed(123)

split <- caTools::sample.split(datos_pca$Customer_Segment, SplitRatio = 0.75)

entrenamiento <- datos_pca %>% 
  subset(split == T)

validacion <- datos_pca %>% 
  subset(split == F)


# ajustar clasificacion ---------------------------------------------------

clasificador <- entrenamiento %>% 
  e1071::svm(Customer_Segment ~ .,
             type = "C-classification",
             kernel = "radial", # el problema no es lineal
             data = .)


# predicción de resultados ------------------------------------------------

y_pred <- predict(clasificador, validacion)

# matriz confusion --------------------------------------------------------

table(validacion$Customer_Segment, y_pred)


# Visualizacion -----------------------------------------------------------

Dim.1 <- seq(min(validacion$Dim.1) - 1,
           max(validacion$Dim.1) + 1, by = 0.05)

Dim.2 <- seq(min(validacion$Dim.2) - 1,
                        max(validacion$Dim.2) + 1, by = 0.05)

tidyr::expand_grid(Dim.1, Dim.2) %>% 
  predict(clasificador, newdata = .) %>% 
  tibble::tibble(tidyr::expand_grid(Dim.1, Dim.2), Customer_Segment = .) %>% 
  ggplot2::ggplot(aes(Dim.1, Dim.2)) + 
  geom_tile(aes(fill = factor(Customer_Segment)), alpha = 0.25) +
  geom_point(data = validacion, 
             aes(Dim.1, Dim.2, color = factor(Customer_Segment))) +
  labs(title = "Clasificador: SVM",
       subtitle = "(Conjunto de test)",
       x = "Dim.1",
       y = "Dim.2",
       fill = NULL,
       color = NULL) +
  scale_fill_brewer(palette = "Dark2") +
  scale_color_brewer(palette = "Dark2") +
  theme_classic()
