## K-means ##


# libraries ---------------------------------------------------------------

library(tidyverse)
library(cluster)


# read data ---------------------------------------------------------------

mall <- readr::read_csv("data/Mall_Customers.csv") %>% 
  janitor::clean_names()

mall <- mall %>% 
  dplyr::select(annual_income_k, spending_score_1_100)


# método del codo ---------------------------------------------------------

set.seed(123)

codo <- function(i){
  
  x <- mall %>% 
    kmeans(i)
  
  wcss <- sum(x$withinss)
  
}

wcss <- c(1:10) %>% 
  purrr::map_dbl(codo)

ggplot2::ggplot(data = NULL, aes(c(1:10), wcss)) + 
  geom_line(color = "darkblue") +
  geom_point(color = "darkblue") +
  labs(title = "Método del codo",
       x = "Número de k's",
       y = "WCSS") +
  theme_classic()


# ajustar algoritmo de cluster --------------------------------------------

cluster <- mall %>% 
  kmeans(5, iter.max = 300, nstart = 10)


# visualización -----------------------------------------------------------

## Ojo, hace un PCA, o similar

cluster::clusplot(mall, cluster$cluster, 
                  lines = 0, shade = TRUE, 
                  color = TRUE)
