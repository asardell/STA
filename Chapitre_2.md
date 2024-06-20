# Lien entre deux variables quantitatives

## Objectifs
Voici les objectifs de ce chapitre :
- [ ] Représenter graphiquement deux variables quantitatives
- [ ] Calculer et interpréter le coefficient de corrélation linéaire
- [ ] Construire une matrice des corrélations
- [ ] Calculer et interpréter le coefficient de détermination

1. [Lien entre deux variables quantitatives](#lien-entre-deux-variables-quantitatives)
   1. [Objectifs](#objectifs)
   2. [Exercice 1 - Coefficient de corrélation linéaire](#exercice-1---coefficient-de-corrélation-linéaire)
      1. [Mémo](#mémo)
      2. [Charger les données.](#charger-les-données)
      3. [Calculer la covariance entre deux variables quantitatives.](#calculer-la-covariance-entre-deux-variables-quantitatives)
      4. [Calculer la covariance manuellement.](#calculer-la-covariance-manuellement)
      5. [Calculer le coefficient de corrélation entre deux variables quantitatives.](#calculer-le-coefficient-de-corrélation-entre-deux-variables-quantitatives)
      6. [Calculer ce coefficient manuellement.](#calculer-ce-coefficient-manuellement)
      7. [Calculer du coefficient de détermination.](#calculer-du-coefficient-de-détermination)
      8. [Construire un nuage de points entre ces deux variables.](#construire-un-nuage-de-points-entre-ces-deux-variables)
      9. [Le Quartet d'Anscombe.](#le-quartet-danscombe)
   3. [Exercice 2 - Matrice de corrélation](#exercice-2---matrice-de-corrélation)
      1. [Calculer la matrice de corrélation des variables quantitatives.](#calculer-la-matrice-de-corrélation-des-variables-quantitatives)

Dans ce chapitre, nous allons utiliser le jeu de données Iris. Il est présent par défaut dans les environnements [R](https://rdrr.io/snippets/) et [Python](https://colab.research.google.com/). Il est aussi accessible dans le classeur Excel de ce repository.

:warning: le dataset peut avoir des différences selon le langage utilisé.

## Exercice 1 - Coefficient de corrélation linéaire

### Mémo
| Nom de l'indicateur | Description    | Notation | Formule                          |
|---------------------|----------------|----------|----------------------------------|
| Covariance   | Mesure de la tendance linéaire entre deux variables aléatoires. | $\text{cov}(X, Y)$ | $\text{cov}(X, Y) = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})$ |
| Coefficient de corrélation | Mesure de la force et de la direction de la relation linéaire entre deux variables aléatoires. | $r$ | $r = \frac{\text{cov}(X, Y)}{s_X \cdot s_Y}$ où $s_X$ et $s_Y$ sont les écart-types de $X$ et $Y$, respectivement. |
| Coefficient de détermination | Mesure la proportion de la variance de la variable dépendante expliquée par la variable indépendante dans un modèle de régression linéaire. | $R^2$ | $R^2 = r^2$ |
| Nuage de points           | Représentation graphique des paires de valeurs $(x_i, y_i)$ pour deux variables $X$ et $Y$. | - | - |

### Charger les données. 
<details>
<summary>R</summary>

```r
# Charger les librairies nécessaires
library(corrplot)

# Charger le jeu de données iris
data(iris)

# Choisir deux variables quantitatives (par exemple, la longueur et la largeur des sépales)
var1 <- "Sepal.Length"
var2 <- "Sepal.Width"
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

# Choisir deux variables quantitatives (par exemple, la longueur et la largeur des sépales)
var1 = 'sepal length (cm)'
var2 = 'sepal width (cm)'
```
</details>

### Calculer la covariance entre deux variables quantitatives. 
<details>
<summary>R</summary>

```r
# Calculer la covariance entre deux variables quantitatives
covariance <- cov(iris[[var1]], iris[[var2]])
cat("Covariance entre", var1, "et", var2, ":", covariance, "\n")
```
</details>

<details>
<summary>Python</summary>

```python
# Calculer la covariance entre deux variables quantitatives
covariance_matrix = iris_df[[var1, var2]].cov()
covariance = covariance_matrix.iloc[0, 1]
print(f"Covariance entre {var1} et {var2}: {covariance}")
```
</details>

<details>
<summary>Excel</summary>

```
```
</details>

### Calculer la covariance manuellement. 
<details>
<summary>R</summary>

```r
# Calculer la covariance manuellement
mean_var1 <- mean(iris[[var1]])
mean_var2 <- mean(iris[[var2]])
covariance_manual <- mean((iris[[var1]] - mean_var1) * (iris[[var2]] - mean_var2))
cat("Covariance manuelle entre", var1, "et", var2, ":", covariance_manual, "\n")
```
</details>

<details>
<summary>Python</summary>

```python
# Calculer la covariance manuellement
mean_var1 = iris_df[var1].mean()
mean_var2 = iris_df[var2].mean()
covariance_manual = np.mean((iris_df[var1] - mean_var1) * (iris_df[var2] - mean_var2))
print(f"Covariance manuelle entre {var1} et {var2}: {covariance_manual}")
```
</details>

<details>
<summary>Excel</summary>

```
```
</details>

### Calculer le coefficient de corrélation entre deux variables quantitatives. 

<details>
<summary>R</summary>

```r
# Calculer le coefficient de corrélation entre deux variables quantitatives
correlation <- cor(iris[[var1]], iris[[var2]])
cat("Coefficient de corrélation entre", var1, "et", var2, ":", correlation, "\n")
```
</details>

<details>
<summary>Python</summary>

```python
# Calculer le coefficient de corrélation entre deux variables quantitatives
correlation_matrix = iris_df[[var1, var2]].corr()
correlation = correlation_matrix.iloc[0, 1]
print(f"Coefficient de corrélation entre {var1} et {var2}: {correlation}")
```
</details>

<details>
<summary>Excel</summary>

```
```
</details>

### Calculer ce coefficient manuellement. 
<details>
<summary>R</summary>

```r
# Calculer le coefficient de corrélation manuellement
std_var1 <- sd(iris[[var1]])
std_var2 <- sd(iris[[var2]])
correlation_manual <- covariance_manual / (std_var1 * std_var2)
cat("Coefficient de corrélation manuel entre", var1, "et", var2, ":", correlation_manual, "\n")
```
</details>

<details>
<summary>Python</summary>

```python
# Calculer le coefficient de corrélation manuellement
std_var1 = iris_df[var1].std()
std_var2 = iris_df[var2].std()
correlation_manual = covariance_manual / (std_var1 * std_var2)
print(f"Coefficient de corrélation manuel entre {var1} et {var2}: {correlation_manual}")
```
</details>

<details>
<summary>Excel</summary>

```
```
</details>

### Calculer du coefficient de détermination. 
<details>
<summary>R</summary>

```r
# Calculer le coefficient de détermination
r_squared <- correlation^2
cat("Coefficient de détermination (R^2) entre", var1, "et", var2, ":", r_squared, "\n")
```
</details>

<details>
<summary>Python</summary>

```python
# Calculer le coefficient de détermination
r_squared = correlation**2
print(f"Coefficient de détermination (R^2) entre {var1} et {var2}: {r_squared}")
```
</details>

<details>
<summary>Excel</summary>

```
```
</details>

### Construire un nuage de points entre ces deux variables. 

[Voici quelques exemples de nuage de points.](https://saylordotorg.github.io/text_introductory-statistics/section_14/07aa5db140b70615a15e8631c2d7a2c4.jpg)

<details>
<summary>R</summary>

```r
# Construire un nuage de points entre les deux variables sélectionnées
plot(iris[[var1]], iris[[var2]], main = paste(var1, " vs ", var2))
```
</details>

<details>
<summary>Python</summary>

```python
# Construire un nuage de points entre ces deux variables
plt.figure(figsize=(10, 6))
sns.scatterplot(data=iris_df, x=var1, y=var2)
plt.title(f"Nuage de points entre {var1} et {var2}")
plt.xlabel(var1)
plt.ylabel(var2)
plt.grid(True)
plt.show()
```
</details>

<details>
<summary>Excel</summary>

```
```
</details>

### Le Quartet d'Anscombe. 

:warning: Il est toujours important de visualiser ses données. Plus d'info avec le [Quartet d'Anscombe](https://blog.revolutionanalytics.com/2017/05/the-datasaurus-dozen.html)

## Exercice 2 - Matrice de corrélation

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

<details>
<summary>Excel</summary>

```
```
</details>