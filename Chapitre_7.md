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
      4. [Lien entre Sepal Width et Sepal Length](#lien-entre-sepal-width-et-sepal-length)
      5. [Lien entre Sepal Length et Petal Width](#lien-entre-sepal-length-et-petal-width)
      6. [Lien entre Sepal Width et Petal Width](#lien-entre-sepal-width-et-petal-width)
   3. [Exercice 2 - Cas de relation non linéaire mais monotone](#exercice-2---cas-de-relation-non-linéaire-mais-monotone)
      1. [Mémo](#mémo-1)
      2. [Simulation une distribution monotome et non linéaire.](#simulation-une-distribution-monotome-et-non-linéaire)
      3. [Effectuer les 3 tests de corrélations.](#effectuer-les-3-tests-de-corrélations)
   4. [Bonus - Cas de relation non linéaire et non monotone](#bonus---cas-de-relation-non-linéaire-et-non-monotone)
      1. [Simulation une distribution non linéaire et non monotone.](#simulation-une-distribution-non-linéaire-et-non-monotone)
      2. [Effectuer les 4 tests de corrélations.](#effectuer-les-4-tests-de-corrélations)
      3. [Mais cela ne fait pas des miracles](#mais-cela-ne-fait-pas-des-miracles)

Dans ce chapitre, nous allons utiliser le jeu de données Iris. Il est présent par défaut dans les environnements [R](https://rdrr.io/snippets/) et [Python](https://colab.research.google.com/). Il est aussi accessible dans le classeur Excel de ce repository.

:warning: le dataset peut avoir des différences selon le langage utilisé.

## Exercice 1 - Coefficient de corrélation linéaire de Pearson

### Mémo
| Nom de l'indicateur | Description    | Notation | Formule                          |
|---------------------|----------------|----------|----------------------------------|
| Coefficient de corrélation  de Pearson | Mesure de la force et de la direction de la relation linéaire entre deux variables aléatoires. | $r$ | $r = \frac{\text{cov}(X, Y)}{s_X \cdot s_Y}$ où $s_X$ et $s_Y$ sont les écart-types de $X$ et $Y$, respectivement. |
| La taille de l'échantillon    | -        | $n$          | -    |
| Degrés de liberté     | Nombre de valeurs libres de varier dans le calcul d'une statistique   | $df$     | $n - 2$      |
| Statistique t                           | Valeur calculée pour tester l'hypothèse nulle dans le test de corrélation    | $t$  | $t = r \sqrt{\frac{n - 2}{1 - r^2}}$                                                               |
| P-value     | Probabilité d'obtenir une statistique t au moins aussi extrême que celle observée, sous l'hypothèse nulle | -            | Déterminée à partir de la distribution t de Student avec $n - 2$ degrés de liberté               |

### Charger les données. 
<details>
<summary>R</summary>

```r
# Charger les librairies nécessaires
library(corrplot)

# Charger le jeu de données Iris depuis le package datasets de R
data(iris)

# Convertir en data frame pour faciliter l'accès
iris_df <- as.data.frame(iris)

# Renommer les colonnes pour faciliter l'accès
colnames(iris_df) <- c('sepal_length', 'sepal_width', 'petal_length', 'petal_width')
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

# Renommer les colonnes pour faciliter l'accès
iris_df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
```
</details>

### Calculer la matrice de corrélation des variables quantitatives. 
<details>
<summary>R</summary>

```r
# Calculer la matrice des corrélations pour les quatre variables quantitatives
library(corrplot)
correlation_matrix_all <- cor(iris_df[, 1:4])
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


### Lien entre Sepal Width et Sepal Length

1. Poser $H_0$ et $H_1$.
- $H_0$ : Les variables sont indépendantes
- $H_1$ : Les variables ne sont pas indépendantes

2. Calculer le coefficient de corrélation linéaire de Pearson.
<details>
<summary>R</summary>

```r
# Sélectionner les colonnes "sepal_width" et "sepal_length"
x <- iris_df$sepal_width
y <- iris_df$sepal_length

# Calculer la covariance entre "sepal_width" et "sepal_length"
covariance <- cov(x, y)

# Calculer les écarts types de "sepal_width" et "sepal_length"
std_x <- sd(x)
std_y <- sd(y)

# Calculer le coefficient de corrélation de Pearson
r_manual <- covariance / (std_x * std_y)
```
</details>

<details>
<summary>Python</summary>

```python
# Sélectionner les colonnes "sepal_width" et "sepal_length"
x = iris_df['sepal_width']
y = iris_df['sepal_length']

# Calculer la covariance entre "sepal_width" et "sepal_length"
covariance = np.cov(x, y, bias=True)[0, 1]

# Calculer les écarts types de "sepal_width" et "sepal_length"
std_x = np.std(x, ddof=0)
std_y = np.std(y, ddof=0)

# Calculer le coefficient de corrélation de Pearson manuellement
r_manual = covariance / (std_x * std_y)
```
</details>

3. Calculer la statistique de test t.
<details>
<summary>R</summary>

```r
# Calculer la statistique de test t
n <- length(x)
df <- n - 2
t_stat <- r_manual * sqrt(df / (1 - r_manual^2))
```
</details>

<details>
<summary>Python</summary>

```python
# Calculer le nombre de degrés de liberté
n = len(x)
df = n - 2

# Calculer la statistique de test t
t_stat = r_manual * np.sqrt(df / (1 - r_manual**2))
```
</details>

4. Calculer la p-value à partir de la loi de Student.
<details>
<summary>R</summary>

```r
# Calculer la p-value manuellement
p_value_manual <- 2 * (1 - pt(abs(t_stat), df))

# Afficher les résultats
print(paste("Coefficient de corrélation (manuel) :", r_manual))
print(paste("Statistique de test t :", t_stat))
print(paste("P-value (manuel) :", p_value_manual))
```
</details>

<details>
<summary>Python</summary>

```python
# Calculer la p-value manuellement
p_value_manual = 2 * (1 - t.cdf(np.abs(t_stat), df))

# Afficher les résultats
print(f"Coefficient de corrélation de Pearson (manuel): {r_manual}")
print(f"Statistique t: {t_stat}")
print(f"Degrés de liberté: {df}")
print(f"P-value (calculée manuellement): {p_value_manual}")
```
</details>

5. Comparer toutes ce résultats avec la fonction test de corrélation.
<details>
<summary>R</summary>

```r
# Utiliser la fonction cor.test pour obtenir le coefficient de corrélation, la statistique t et la p-value
cor_test <- cor.test(x, y)
r_cor_test <- cor_test$estimate
t_stat_cor_test <- cor_test$statistic
p_value_cor_test <- cor_test$p.value

# Afficher les résultats de cor.test
print(paste("Coefficient de corrélation (cor.test) :", r_cor_test))
print(paste("Statistique de test t (cor.test) :", t_stat_cor_test))
print(paste("P-value (cor.test) :", p_value_cor_test))
```
</details>

<details>
<summary>Python</summary>

```python
# Vérifier avec la fonction scipy.stats.pearsonr
from scipy.stats import pearsonr
r_scipy, p_value_scipy = pearsonr(x, y)
print(f"Coefficient de corrélation de Pearson (scipy): {r_scipy}")
print(f"P-value (scipy): {p_value_scipy}")

# Comparer les valeurs
print("\nComparaison des valeurs calculées manuellement et avec scipy.stats.pearsonr:")
print(f"Coefficient de corrélation: {r_manual} (manuel) vs {r_scipy} (scipy)")
print(f"P-value: {p_value_manual} (manuel) vs {p_value_scipy} (scipy)")
```
</details>

6. Que peut-on affirmer avec un risque de se tromper $\alpha = 0.05$ ?

Les deux variables sont indépendantes.

### Lien entre Sepal Length et Petal Width

1. Poser $H_0$ et $H_1$.
- $H_0$ : Les variables sont indépendantes
- $H_1$ : Les variables ne sont pas indépendantes

2. Effectuer le test de corrélation.
<details>
<summary>R</summary>

```r
x <- iris_df$sepal_length
y <- iris_df$petal_width
cor_test <- cor.test(x, y)
r_cor_test <- cor_test$estimate
t_stat_cor_test <- cor_test$statistic
p_value_cor_test <- cor_test$p.value

# Afficher les résultats de cor.test
print(paste("Coefficient de corrélation (cor.test) :", r_cor_test))
print(paste("Statistique de test t (cor.test) :", t_stat_cor_test))
print(paste("P-value (cor.test) :", p_value_cor_test))
```
</details>

<details>
<summary>Python</summary>

```python
# Sélectionner les colonnes "sepal_width" et "sepal_length"
x = iris_df['sepal_length']
y = iris_df['petal_width']
r_scipy, p = pearsonr(x, y)
print(f"Coefficient de corrélation: {r_scipy}")
print(f"P-value: {p}")
```
</details>

3. Que peut-on affirmer avec un risque de se tromper $\alpha = 0.05$ ?

Les deux variables ne sont pas indépendantes.

### Lien entre Sepal Width et Petal Width

1. Poser $H_0$ et $H_1$.
- $H_0$ : Les variables sont indépendantes
- $H_1$ : Les variables ne sont pas indépendantes


2. Effectuer le test de corrélation.
<details>
<summary>R</summary>

```r
x <- iris_df$sepal_width
y <- iris_df$petal_width
cor_test <- cor.test(x, y)
r_cor_test <- cor_test$estimate
t_stat_cor_test <- cor_test$statistic
p_value_cor_test <- cor_test$p.value

# Afficher les résultats de cor.test
print(paste("Coefficient de corrélation (cor.test) :", r_cor_test))
print(paste("Statistique de test t (cor.test) :", t_stat_cor_test))
print(paste("P-value (cor.test) :", p_value_cor_test))
```
</details>

<details>
<summary>Python</summary>

```python
# Sélectionner les colonnes "sepal_width" et "sepal_length"
x = iris_df['sepal_width']
y = iris_df['petal_width']
r_scipy, p = pearsonr(x, y)
print(f"Coefficient de corrélation: {r_scipy}")
print(f"P-value: {p}")
```
</details>

3. Que peut-on affirmer avec un risque de se tromper $\alpha = 0.05$ ?

Les deux variables ne sont pas indépendantes.

## Exercice 2 - Cas de relation non linéaire mais monotone

### Mémo
| Nom de l'indicateur                     | Description | Notation | Formule |
|-----------------------------------------|-------------|----------|---------|
| **Coefficient de corrélation de Pearson** | Mesure de la force et de la direction de la relation linéaire entre deux variables aléatoires. | $r$ | $r = \frac{\text{cov}(X, Y)}{s_X \cdot s_Y}$ où $s_X$ et $s_Y$ sont les écart-types de $X$ et $Y$, respectivement. |
| **Coefficient de corrélation de Kendall** | Mesure de la force et de la direction de la relation monotone entre deux variables aléatoires. | $\tau$ | $\tau = \frac{n_c - n_d}{\frac{1}{2}n(n-1)}$ où $n_c$ est le nombre de paires concordantes et $n_d$ est le nombre de paires discordantes. |
| **Coefficient de corrélation de Spearman** | Mesure de la force et de la direction de la relation monotone entre deux variables aléatoires, basé sur les rangs des données. | $\rho$ | $\rho = 1 - \frac{6 \sum d_i^2}{n(n^2 - 1)}$ où $d_i$ est la différence entre les rangs des paires de données. |
| **La taille de l'échantillon** | - | $n$ | - |
| **Degrés de liberté (Pearson et Spearman)** | Nombre de valeurs libres de varier dans le calcul d'une statistique | $df$ | $n - 2$ |
| **Statistique t (Pearson)** | Valeur calculée pour tester l'hypothèse nulle dans le test de corrélation | $t$ | $t = r \sqrt{\frac{n - 2}{1 - r^2}}$ |
| **Statistique z (Kendall)** | Valeur calculée pour tester l'hypothèse nulle dans le test de corrélation de Kendall | $z$ | $z = \frac{\tau \sqrt{9n(n-1)}}{\sqrt{2(2n+5)}}$ |
| **Statistique t (Spearman)** | Valeur calculée pour tester l'hypothèse nulle dans le test de corrélation de Spearman | $t$ | $t = \rho \sqrt{\frac{n - 2}{1 - \rho^2}}$ |
| **P-value (Pearson)** | Probabilité d'obtenir une statistique t au moins aussi extrême que celle observée, sous l'hypothèse nulle | - | Déterminée à partir de la distribution t de Student avec $n - 2$ degrés de liberté |
| **P-value (Kendall)** | Probabilité d'obtenir une statistique z au moins aussi extrême que celle observée, sous l'hypothèse nulle | - | Déterminée à partir de la distribution normale standard |
| **P-value (Spearman)** | Probabilité d'obtenir une statistique t au moins aussi extrême que celle observée, sous l'hypothèse nulle | - | Déterminée à partir de la distribution t de Student avec $n - 2$ degrés de liberté |

### Simulation une distribution monotome et non linéaire. 
<details>
<summary>R</summary>

```r
# Simuler une distribution non linéaire mais monotone avec plus de perturbation gaussienne
set.seed(0)
x <- runif(100, -1.5, 1.5)
y <- sign(x) * (x^4 + 0.1 * rnorm(length(x)))

# Visualiser les données
df <- data.frame(x = x, y = y)
ggplot(df, aes(x = x, y = y)) + 
  geom_point() + 
  ggtitle("Distribution non linéaire mais monotone avec plus de perturbation") +
  xlab("x") +
  ylab("y")
```
</details>

<details>
<summary>Python</summary>

```python
import numpy as np
import matplotlib.pyplot as plt

# Simuler une distribution non linéaire mais monotone
np.random.seed(0)
x = np.random.uniform(-1.5, 1.5, 100)
y = np.sign(x) * (x**4 + 0.01 * np.random.normal(size=x.size))

# Visualiser les données
plt.scatter(x, y)
plt.title('Distribution non linéaire mais monotone')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
```
</details>

### Effectuer les 3 tests de corrélations. 
<details>
<summary>R</summary>

```r
# Calculer les coefficients de corrélation
pearson_corr <- cor.test(x, y, method = "pearson")
spearman_corr <- cor.test(x, y, method = "spearman")
kendall_corr <- cor.test(x, y, method = "kendall")

# Afficher les résultats
cat("Coefficient de corrélation de Pearson: ", pearson_corr$estimate, ", p-value: ", pearson_corr$p.value, "\n")
cat("Coefficient de corrélation de Spearman: ", spearman_corr$estimate, ", p-value: ", spearman_corr$p.value, "\n")
cat("Coefficient de corrélation de Kendall: ", kendall_corr$estimate, ", p-value: ", kendall_corr$p.value, "\n")
```
</details>

<details>
<summary>Python</summary>

```python
from scipy.stats import pearsonr, spearmanr, kendalltau
# Calculer les coefficients de corrélation
pearson_corr, pearson_p = pearsonr(x, y)
spearman_corr, spearman_p = spearmanr(x, y)
kendall_corr, kendall_p = kendalltau(x, y)

# Afficher les résultats
print(f"Coefficient de corrélation de Pearson: {pearson_corr:.4f}, p-value: {pearson_p:.4f}")
print(f"Coefficient de corrélation de Spearman: {spearman_corr:.4f}, p-value: {spearman_p:.4f}")
print(f"Coefficient de corrélation de Kendall: {kendall_corr:.4f}, p-value: {kendall_p:.4f}")
```
</details>


## Bonus - Cas de relation non linéaire et non monotone

### Simulation une distribution non linéaire et non monotone. 
<details>
<summary>R</summary>

```r
```
</details>

<details>
<summary>Python</summary>

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, kendalltau
from minepy import MINE

# Simuler une distribution non linéaire et non monotone avec du bruit gaussien
np.random.seed(0)
x = np.random.uniform(0, 1, 100)
y = np.sin(4 * np.pi * x) + 0.1 * np.random.normal(size=x.size)

# Visualiser les données
plt.scatter(x, y)
plt.title('Distribution non linéaire et non monotone')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
```
</details>

### Effectuer les 4 tests de corrélations. 
<details>
<summary>R</summary>

```r
```
</details>

<details>
<summary>Python</summary>

```python
# Calculer les coefficients de corrélation
pearson_corr, pearson_p = pearsonr(x, y)
spearman_corr, spearman_p = spearmanr(x, y)
kendall_corr, kendall_p = kendalltau(x, y)

# Calculer le MIC avec MINE
mine = MINE()
mine.compute_score(x, y)
mic = mine.mic()

# Afficher les résultats
print(f"Coefficient de corrélation de Pearson: {pearson_corr:.4f}, p-value: {pearson_p:.4f}")
print(f"Coefficient de corrélation de Spearman: {spearman_corr:.4f}, p-value: {spearman_p:.4f}")
print(f"Coefficient de corrélation de Kendall: {kendall_corr:.4f}, p-value: {kendall_p:.4f}")
print(f"Maximum Information Coefficient (MIC): {mic:.4f}")
```
</details>


### Mais cela ne fait pas des miracles

<details>
<summary>Python</summary>

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, kendalltau
from minepy import MINE
from sklearn.datasets import make_regression, make_circles

# Simuler une distribution non linéaire et non monotone avec du bruit gaussien
np.random.seed(0)

X = make_circles(n_samples=100,factor=0.99, random_state=0, noise=0.05)[0]
X = pd.DataFrame(X, columns=['x','y'])

x = X['x']
y = X['y']

# Visualiser les données
plt.scatter(x, y)
plt.title('Distribution non linéaire et non monotone')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# Calculer les coefficients de corrélation
pearson_corr, pearson_p = pearsonr(x, y)
spearman_corr, spearman_p = spearmanr(x, y)
kendall_corr, kendall_p = kendalltau(x, y)

# Calculer le MIC avec MINE
mine = MINE()
mine.compute_score(x, y)
mic = mine.mic()

# Afficher les résultats
print(f"Coefficient de corrélation de Pearson: {pearson_corr:.4f}, p-value: {pearson_p:.4f}")
print(f"Coefficient de corrélation de Spearman: {spearman_corr:.4f}, p-value: {spearman_p:.4f}")
print(f"Coefficient de corrélation de Kendall: {kendall_corr:.4f}, p-value: {kendall_p:.4f}")
print(f"Maximum Information Coefficient (MIC): {mic:.4f}")
```
</details>