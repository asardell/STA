# Lien entre une variable qualitative et quantitative

## Objectifs
Voici les objectifs de ce chapitre :
- [ ] Représenter graphiquement avec un boxplot
- [ ] Calculer et interpréter les moyennes marginales
- [ ] Calculer et interpréter les variances marginales

1. [Lien entre une variable qualitative et quantitative](#lien-entre-une-variable-qualitative-et-quantitative)
   1. [Objectifs](#objectifs)
   2. [Exercice 1 - Description des groupes](#exercice-1---description-des-groupes)
      1. [Mémo](#mémo)
      2. [Charger les données.](#charger-les-données)
      3. [Calculer des effectifs de chaque groupe.](#calculer-des-effectifs-de-chaque-groupe)
      4. [Calculer des moyennes (générale et marginales).](#calculer-des-moyennes-générale-et-marginales)
      5. [Calculer des variances (générale et marginales).](#calculer-des-variances-générale-et-marginales)
      6. [Représenter graphiquement avec un boxplot.](#représenter-graphiquement-avec-un-boxplot)

Dans ce chapitre, nous allons utiliser le jeu de données Iris. Il est présent par défaut dans les environnements [R](https://rdrr.io/snippets/) et [Python](https://colab.research.google.com/). Il est aussi accessible dans le classeur Excel de ce repository.

:warning: le dataset peut avoir des différences selon le langage utilisé.

## Exercice 1 - Description des groupes

:warning: Dans cet exercice, on choisit d'étudier l'espèce de fleur en fonction de la longueur de Sepal.

### Mémo
| Nom de l'indicateur | Description    | Notation | Formule                          |
|---------------------|----------------|----------|----------------------------------|
| Moyenne générale           | La moyenne de toutes les observations dans l'ensemble de données.           | $\bar{x}$        | $\bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i$  |
| Variance générale          | La variance de toutes les observations dans l'ensemble de données.          | $s^2$            | $s^2 = \frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{x})^2$   |
| Moyenne marginale          | La moyenne des observations pour chaque niveau d'un facteur.              | $\bar{x}_i$    | $\bar{x_i} = \frac{1}{n_i} \sum_{j=1}^{n_i} x_{ij}$ |
| Variance marginale         | La variance des observations pour chaque niveau d'un facteur.               | $s^2_i$        | $s^2_i = \frac{1}{n_i} \sum_{j=1}^{n_i} (x_{ij} - \bar{x}_i)^2$           |

### Charger les données. 
<details>
<summary>R</summary>

```r
# Charger les librairies nécessaires
library(dplyr)
library(ggplot2)

# Charger le jeu de données Iris
data(iris)
```
</details>

<details>
<summary>Python</summary>

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Charger le jeu de données Iris
from sklearn.datasets import load_iris
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Renommer les colonnes pour faciliter l'utilisation
iris_df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
```
</details>

### Calculer des effectifs de chaque groupe. 

<details>
<summary>R</summary>

```r
# Calculer des effectifs de chaque espèce
species_counts <- iris %>% count(Species)
cat("Effectifs de chaque espèce:\n")
print(species_counts)
```
</details>

<details>
<summary>Python</summary>

```python
# Calculer des effectifs de chaque espèce
species_counts = iris_df['species'].value_counts()
print("Effectifs de chaque espèce:")
print(species_counts)
```
</details>

<details>
<summary>Excel</summary>

```
```
</details>

### Calculer des moyennes (générale et marginales). 

<details>
<summary>R</summary>

```r
# Calculer des moyennes (générale et marginales)
general_mean <- mean(iris$Sepal.Length)
species_means <- iris %>% group_by(Species) %>% summarise(mean_sepal_length = mean(Sepal.Length))
cat("\nMoyenne générale de la longueur de Sepal:\n")
print(general_mean)
cat("Moyennes marginales de la longueur de Sepal par espèce:\n")
print(species_means)
```
</details>

<details>
<summary>Python</summary>

```python
# Calculer des moyennes (générale et marginales)
general_mean = iris_df['sepal_length'].mean()
species_means = iris_df.groupby('species')['sepal_length'].mean()
print("\nMoyenne générale de la longueur de Sepal:")
print(general_mean)
print("Moyennes marginales de la longueur de Sepal par espèce:")
print(species_means)
```
</details>

<details>
<summary>Excel</summary>

```
```
</details>

### Calculer des variances (générale et marginales). 

<details>
<summary>R</summary>

```r
# Calculer des variances (générale et marginales)
general_variance <- var(iris$Sepal.Length)
species_variances <- iris %>% group_by(Species) %>% summarise(var_sepal_length = var(Sepal.Length))
cat("\nVariance générale de la longueur de Sepal:\n")
print(general_variance)
cat("Variances marginales de la longueur de Sepal par espèce:\n")
print(species_variances)
```
</details>

<details>
<summary>Python</summary>

```python
# Calculer des variances (générale et marginales)
general_variance = iris_df['sepal_length'].var()
species_variances = iris_df.groupby('species')['sepal_length'].var()
print("\nVariance générale de la longueur de Sepal:")
print(general_variance)
print("Variances marginales de la longueur de Sepal par espèce:")
print(species_variances)
```
</details>

<details>
<summary>Excel</summary>

```
```
</details>

### Représenter graphiquement avec un boxplot. 

<details>
<summary>R</summary>

```r
# Représenter graphiquement avec un boxplot
ggplot(iris, aes(x = Species, y = Sepal.Length)) +
  geom_boxplot() +
  labs(title = "Boxplot de la longueur de Sepal par espèce",
       x = "Espèce",
       y = "Longueur de Sepal (cm)") +
  theme_minimal()
```
</details>

<details>
<summary>Python</summary>

```python
# Représenter graphiquement avec un boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x='species', y='sepal_length', data=iris_df)
plt.title("Boxplot de la longueur de Sepal par espèce")
plt.xlabel("Espèce")
plt.ylabel("Longueur de Sepal (cm)")
plt.show()
```
</details>

<details>
<summary>Excel</summary>

```
```
</details>


:bulb: On peut aussi tester les liens avec les autres variables quantitatives.