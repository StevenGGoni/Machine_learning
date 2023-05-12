## Cluster jerarquico ##


# libraries ---------------------------------------------------------------

library(tidyverse)
library(cluster)
library(ggdendro)

# read data ---------------------------------------------------------------

mall <- readr::read_csv("data/Mall_Customers.csv") %>% 
  janitor::clean_names()

mall <- mall %>% 
  dplyr::select(annual_income_k, spending_score_1_100)


# dendrograma -------------------------------------------------------------

dend <- dist(mall, method = "euclidean") %>% 
  hclust(method = "ward.D")

plot(dend)

ggdendro::ggdendrogram(dend) +
  labs(title = "Dendograma",
       x =  "Clientes",
       y = "Distancia") +
  coord_flip() +
  theme_classic()

# ajustar algoritmo de cluster --------------------------------------------

cluster <- dist(mall, method = "euclidean") %>% 
  hclust(method = "ward.D")

y_cluster <- cutree(cluster, 5)

# visualización -----------------------------------------------------------

## Ojo, hace un PCA, o similar

cluster::clusplot(mall, y_cluster, 
                  lines = 0, shade = TRUE, 
                  color = TRUE)
