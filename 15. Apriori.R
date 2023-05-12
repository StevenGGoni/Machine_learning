## Apriori ##


# libraries ---------------------------------------------------------------

library(tidyverse)
library(arules)
library(arulesViz)


# read data ---------------------------------------------------------------

data <- arules::read.transactions("data/Market_Basket_Optimisation.csv", 
                                  sep = ",", rm.duplicates = TRUE)

summary(data)

arules::itemFrequencyPlot(data, topN = 10) # 10 productos más comprados

# para hacer el gráfico bonito, puedo acceder a los datos así:
frec <- arules::itemFrequency(data) %>% 
  as.data.frame() %>% 
  tibble::rownames_to_column()

# entrenar algoritmo ------------------------------------------------------

rules <- arules::apriori(data, 
                         parameter = list(support = 0.004, confidence = 0.2))

# Reglas fuertes
# Visualización de resultados

arules::inspect(arules::sort(rules, by = "lift")[1:10])

plot(rules, method = "graph", engine = "htmlwidget")
