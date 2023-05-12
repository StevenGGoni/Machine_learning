## LDA ##


# libraries ---------------------------------------------------------------

library(tidyverse)
library(MASS)


# read data ---------------------------------------------------------------

datos <- readr::read_csv("data/Wine.csv")

# escalado de variables ---------------------------------------------------

datos <- datos %>% 
  dplyr::mutate(across(1:13, scale)) %>% 
  dplyr::mutate(across(1:13, as.numeric))

# lda ---------------------------------------------------------------------

lda <- MASS::lda(Customer_Segment ~ ., datos) %>% 
  predict(datos)

datos_lda <- data.frame(lda$x, lda$class) %>% 
  dplyr::rename(Customer_Segment = lda.class)

# Entrenamiento y validación ----------------------------------------------

set.seed(123)

split <- caTools::sample.split(datos_lda$Customer_Segment, SplitRatio = 0.75)

entrenamiento <- datos_lda %>% 
  subset(split == T)

validacion <- datos_lda %>% 
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

LD1 <- seq(min(validacion$LD1) - 1,
             max(validacion$LD1) + 1, by = 0.05)

LD2 <- seq(min(validacion$LD2) - 1,
             max(validacion$LD2) + 1, by = 0.05)

tidyr::expand_grid(LD1, LD2) %>% 
  predict(clasificador, newdata = .) %>% 
  tibble::tibble(tidyr::expand_grid(LD1, LD2), Customer_Segment = .) %>% 
  ggplot2::ggplot(aes(LD1, LD2)) + 
  geom_tile(aes(fill = factor(Customer_Segment)), alpha = 0.25) +
  geom_point(data = validacion, 
             aes(LD1, LD2, color = factor(Customer_Segment))) +
  labs(title = "Clasificador: SVM",
       subtitle = "(Conjunto de test)",
       x = "LD1",
       y = "LD2",
       fill = NULL,
       color = NULL) +
  scale_fill_brewer(palette = "Dark2") +
  scale_color_brewer(palette = "Dark2") +
  theme_classic()

