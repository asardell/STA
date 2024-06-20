# Le test de corrélation

## Objectifs
Voici les objectifs de ce chapitre :
- [ ] Calculer et interpréter un test de corrélation de Pearson
- [ ] Différencier un test paramétrique et non paramétrique 
- [ ] Analyser la p-value
- [ ] Comprendre l'intérêt des hypothèses de départ et alternative

1. [Le test de corrélation](#le-test-de-corrélation)
   1. [Objectifs](#objectifs)
   2. [Exercice 1 - Coefficient de corrélation linéaire de Pearson](#exercice-1---coefficient-de-corrélation-linéaire-de-pearson)
      1. [Mémo](#mémo)
      2. [Charger les données.](#charger-les-données)
      3. [Calculer la matrice de corrélation des variables quantitatives.](#calculer-la-matrice-de-corrélation-des-variables-quantitatives)
      4. [Lien entre Sepal Length et Petal Width](#lien-entre-sepal-length-et-petal-width)
      5. [Lien entre Sepal Width et Sepal Length](#lien-entre-sepal-width-et-sepal-length)
      6. [Lien entre Sepal Width et Petal Width](#lien-entre-sepal-width-et-petal-width)
   3. [Exercice 3 - L'influence de la taille de l'échantillon](#exercice-3---linfluence-de-la-taille-de-léchantillon)
   4. [Exercice 4 - Cas de relation non linéaire mais monotone](#exercice-4---cas-de-relation-non-linéaire-mais-monotone)
   5. [Bonus - Cas de relation non linéaire et non monotone](#bonus---cas-de-relation-non-linéaire-et-non-monotone)

Dans ce chapitre, nous allons utiliser le jeu de données Iris. Il est présent par défaut dans les environnements [R](https://rdrr.io/snippets/) et [Python](https://colab.research.google.com/). Il est aussi accessible dans le classeur Excel de ce repository.

:warning: le dataset peut avoir des différences selon le langage utilisé.

## Exercice 1 - Coefficient de corrélation linéaire de Pearson

### Mémo
| Nom de l'indicateur | Description    | Notation | Formule                          |
|---------------------|----------------|----------|----------------------------------|
| Coefficient de corrélation | Mesure de la force et de la direction de la relation linéaire entre deux variables aléatoires. | $r$ | $r = \frac{\text{cov}(X, Y)}{s_X \cdot s_Y}$ où $s_X$ et $s_Y$ sont les écart-types de $X$ et $Y$, respectivement. |
| La taille de l'échantillon    | -        | $n$          | -    |
| Degrés de liberté     | Nombre de valeurs libres de varier dans le calcul d'une statistique   | $df$     | $n - 2$      |
| Coefficient de corrélation de Pearson   | Mesure de la force et de la direction de la relation linéaire entre deux variables | $r$          | $r = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum (x_i - \bar{x})^2 \sum (y_i - \bar{y})^2}}$ |
| Statistique t                           | Valeur calculée pour tester l'hypothèse nulle dans le test de corrélation    | $t$  | $t = r \sqrt{\frac{n - 2}{1 - r^2}}$                                                               |
| P-value     | Probabilité d'obtenir une statistique t au moins aussi extrême que celle observée, sous l'hypothèse nulle | -            | Déterminée à partir de la distribution t de Student avec $n - 2$ degrés de liberté               |
| Coefficient de corrélation de Pearson | Mesure de la force et de la direction de la relation linéaire entre deux variables continues. | $r_P$                | $r_P = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum (x_i - \bar{x})^2 \sum (y_i - \bar{y})^2}}$    |
| Coefficient de corrélation de Spearman | Mesure de la force et de la direction de la relation monotone entre deux variables ordinales ou continues. | $\rho$                | $\rho = 1 - \frac{6 \sum d_i^2}{n(n^2 - 1)}$ où $d_i$ est la différence entre les rangs de chaque paire de données. |
| Coefficient de corrélation de Kendall | Mesure de la force et de la direction de la relation monotone entre deux variables ordinales ou continues. | $\tau$                | $\tau = \frac{2 (C - D)}{n(n-1)}$ où $C$ est le nombre de paires concordantes et $D$ le nombre de paires discordantes. |
| Coefficient de Corrélation Maximal d'Information (MIC) | Mesure la force d'association entre deux variables, capable de capturer une large gamme de types de relations, y compris les relations linéaires, non linéaires, monotones et non monotones. | $MIC$ | Calculé via des méthodes d'information mutuelle. Par exemple, avec l'algorithme MINE. |

### Charger les données. 
<details>
<summary>R</summary>

```r
# Charger les librairies nécessaires
library(corrplot)

# Charger le jeu de données iris
data(iris)
```
</details>

<details>
<summary>Python</summary>

```python
from sklearn import datasets
import pandas as pd

# Charger le jeu de données Iris depuis sklearn
iris = datasets.load_iris()

# Convertir en DataFrame pandas pour faciliter l'affichage
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
```
</details>

### Calculer la matrice de corrélation des variables quantitatives. 
<details>
<summary>R</summary>

```r
# Calculer la matrice des corrélations pour les quatre variables quantitatives
library(corrplot)
correlation_matrix_all <- cor(iris[, 1:4])
cat("Matrice des corrélations pour les quatre variables quantitatives :\n")
print(correlation_matrix_all)

# Visualiser la matrice des corrélations sous forme de heatmap
corrplot(correlation_matrix_all, method = "color", addCoef.col = "black", tl.col = "black", 
         tl.srt = 45, title = "Matrice des corrélations des variables quantitatives de l'iris")

```
</details>

<details>
<summary>Python</summary>

```python
# Calculer la matrice des corrélations pour les quatre variables quantitatives
correlation_matrix_all = iris_df.corr()
print("Matrice des corrélations pour les quatre variables quantitatives :")
print(correlation_matrix_all)

# Visualiser la matrice des corrélations à l'aide d'une heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix_all, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Matrice des corrélations des variables quantitatives de l'iris")
plt.show()
```
</details>


### Lien entre Sepal Length et Petal Width

1. Poser $H_0$ et $H_1$.
- $H_0$ : Les variables sont indépendantes
- $H_1$ : Les variables ne sont pas indépendantes

2. .
<details>
<summary>R</summary>

```r

```
</details>

<details>
<summary>Python</summary>

```python

```
</details>

1. xxx.
<details>
<summary>R</summary>

```r

```
</details>

<details>
<summary>Python</summary>

```python

```
</details>

1. xxx.
<details>
<summary>R</summary>

```r

```
</details>

<details>
<summary>Python</summary>

```python

```
</details>

1. xxx.
<details>
<summary>R</summary>

```r

```
</details>

<details>
<summary>Python</summary>

```python

```
</details>

### Lien entre Sepal Width et Sepal Length

### Lien entre Sepal Width et Petal Width

## Exercice 3 - L'influence de la taille de l'échantillon

## Exercice 4 - Cas de relation non linéaire mais monotone

## Bonus - Cas de relation non linéaire et non monotone


#longueur sepal largeur petal