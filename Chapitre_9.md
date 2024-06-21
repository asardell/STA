# L'analyse de la variance

## Objectifs
Voici les objectifs de ce chapitre :
- [ ] Calculer et interpréter le tableau d'analyse de la variance
- [ ] Analyser la p-value
- [ ] Comprendre l'intérêt des hypothèses de départ et alternative
- [ ] Connaître les conditions de validités

1. [L'analyse de la variance](#lanalyse-de-la-variance)
   1. [Objectifs](#objectifs)
   2. [Exercice 1 : Cas théorique](#exercice-1--cas-théorique)
      1. [Mémo](#mémo)
   3. [Exercice 2 : Cas Pratique](#exercice-2--cas-pratique)
      1. [Lien entre Species et Sepal Length](#lien-entre-species-et-sepal-length)
      2. [Lien entre Species et Sepal Width](#lien-entre-species-et-sepal-width)
      3. [Lien entre Species et Petal Width](#lien-entre-species-et-petal-width)
      4. [Lien entre Species et Petal Length](#lien-entre-species-et-petal-length)
   4. [Aller plus loin :  Conditions de validité de l'ANOVA](#aller-plus-loin---conditions-de-validité-de-lanova)

## Exercice 1 : Cas théorique

### Mémo
| Nom de l'indicateur | Description    | Notation | Formule                          |
|---------------------|----------------|----------|----------------------------------|
| **Somme des carrés totale (SST)**           | Mesure de la variabilité totale dans les données      | SST       | $\text{SST} = \sum (X_{ij} - \bar{X})^2$ où $X_{ij}$ est chaque valeur et $\bar{X}$ est la moyenne globale.      |
| **Somme des carrés entre les groupes (SSB)** | Mesure de la variabilité entre les groupes            | SSB       | $\text{SSB} = \sum n_j (\bar{X}_j - \bar{X})^2$ où $n_j$ est la taille du groupe $j$, $\bar{X}_j$ est la moyenne du groupe $j$, et $\bar{X}$ est la moyenne globale. |
| **Somme des carrés à l'intérieur des groupes (SSW)** | Mesure de la variabilité à l'intérieur des groupes                                                  | SSW       | $\text{SSW} = \sum_{j=1}^{k} \sum_{i=1}^{n_j} (X_{ij} - \bar{X}_j)^2$ où $X_{ij}$ est chaque valeur et $\bar{X}_j$ est la moyenne du groupe $j$.   |
| **Degrés de liberté entre les groupes**     | Nombre de valeurs libres de varier entre les groupes     | dfB       | $dfB = k - 1$ où $k$ est le nombre de groupes.      |
| **Degrés de liberté à l'intérieur des groupes** | Nombre de valeurs libres de varier à l'intérieur des groupes        | dfW       | $dfW = n - k$ où $n$ est le nombre total d'observations et $k$ est le nombre de groupes.      |
| **Moyenne des carrés entre les groupes (MSB)** | Mesure de la variabilité moyenne entre les groupes      | MSB    | $\text{MSB} = \frac{\text{SSB}}{dfB}$     |
| **Moyenne des carrés à l'intérieur des groupes (MSW)** | Mesure de la variabilité moyenne à l'intérieur des groupes   | MSW   | $\text{MSW} = \frac{\text{SSW}}{dfW}$      |
| **Statistique F**      | Valeur calculée pour tester l'hypothèse nulle dans le test ANOVA      | F         | $F = \frac{\text{MSB}}{\text{MSW}}$    |
| **P-value**     | Probabilité d'obtenir une statistique F au moins aussi extrême que celle observée, sous l'hypothèse nulle | -   | Déterminée à partir de la distribution F avec $dfB$ et $dfW$ degrés de liberté                                                                                     |

:bulb: On peut résumer ces informations dans le [tableau](https://cdn1.byjus.com/wp-content/uploads/2020/09/one-way-ANOVA-formulas.png) de décomposition de la variance.

L'ANOVA est un test statistique où l'on étudie les variances de chaque groupe pour affirmer s'il y a des moyennes significativement différementes. On pose donc les hypothèses : 

- $H_0$ : Les deux variables sont indépendantes car toutes les moyennes de chaque groupe sont égales 
- $H_1$ : Les deux variables ne sont pas indépendantes car il y a au moins une moyenne d'un groupe différente

1. Simuler les donnée ci-dessous.

<details>
<summary>R</summary>

```r
library(ggplot2)
library(dplyr)
library(car)

# Données
A <- seq(18, 20, length.out = 10)
B <- seq(17, 19, length.out = 10)
C <- seq(17, 21, length.out = 10)
Groupe <- rep(c("A", "B", "C"), each = 10)
Valeur <- c(A, B, C)

result <- data.frame(Groupe = Groupe, Valeur = Valeur)

# Boxplot
ggplot(result, aes(x = Groupe, y = Valeur)) +
  geom_boxplot() +
  theme_minimal()

# Calculer les moyennes et variances
mean_values <- result %>% group_by(Groupe) %>% summarise(mean = mean(Valeur))
var_values <- result %>% group_by(Groupe) %>% summarise(var = var(Valeur))
print(mean_values)
print(var_values)
```
</details>

<details>
<summary>Python</summary>

```python
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd

from scipy.stats import bartlett, shapiro
from scipy.stats import f

import statsmodels.stats.multicomp as multi 
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Données
A = pd.Series(np.linspace(18,20,10), name='A')
B = pd.Series(np.linspace(17,19,10), name='B')
C = pd.Series(np.linspace(17,21,10), name='C')
Groupe = pd.Series(['A', 'B', 'C']).repeat(10).to_list()

frame = { 'Groupe': Groupe, 'Valeur': pd.concat([A, B, C]) } 
result = pd.DataFrame(frame) 
plt.subplots(figsize=(20,4))
ax = sns.boxplot(x="Valeur", y="Groupe", data=result)

print(result.groupby("Groupe")['Valeur'].agg('mean'))
print(result.groupby("Groupe")['Valeur'].agg('var'))
```
</details>

2. Calculer SST, SSB et SSW.

<details>
<summary>R</summary>

```r
# Somme totale des carrés (SST)
grand_mean <- mean(result$Valeur)
sst <- sum((result$Valeur - grand_mean)^2)

# Somme des carrés entre les groupes (SSB)
ssb <- sum(table(result$Groupe) * (tapply(result$Valeur, result$Groupe, mean) - grand_mean)^2)

# Somme des carrés à l'intérieur des groupes (SSW)
ssw <- sum((result$Valeur - ave(result$Valeur, result$Groupe, FUN = mean))^2)
```
</details>

<details>
<summary>Python</summary>

```python
# Somme totale des carrés (SST)
grand_mean = result['Valeur'].mean()
sst = ((result['Valeur'] - grand_mean)**2).sum()

# Somme des carrés entre les groupes (SSB)
ssb = sum(result.groupby('Groupe').size() * (result.groupby('Groupe')['Valeur'].mean() - grand_mean)**2)

# Somme des carrés à l'intérieur des groupes (SSW)
ssw = sum((result['Valeur'] - result.groupby('Groupe')['Valeur'].transform('mean'))**2)

```
</details>

3. Calculer les degrés de liberté.

<details>
<summary>R</summary>

```r
# Degrés de liberté
dfb <- length(unique(result$Groupe)) - 1
dfw <- nrow(result) - length(unique(result$Groupe))

```
</details>

<details>
<summary>Python</summary>

```python
# Degrés de liberté
dfb = len(result['Groupe'].unique()) - 1
dfw = result.shape[0] - len(result['Groupe'].unique())
```
</details>

4. Calculer la moyenne des carrés.

<details>
<summary>R</summary>

```r
# Moyenne des carrés
msb <- ssb / dfb
msw <- ssw / dfw
```
</details>

<details>
<summary>Python</summary>

```python
# Moyenne des carrés
msb = ssb / dfb
msw = ssw / dfw
```
</details>

5. Calculer la statistique F.

<details>
<summary>R</summary>

```r
# Statistique F
f_stat <- msb / msw
```
</details>

<details>
<summary>Python</summary>

```python
# Statistique F
f_stat = msb / msw
```
</details>

6. Calculer et interpréter la p-value.

<details>
<summary>R</summary>

```r
# p-value
p_value <- 1 - pf(f_stat, dfb, dfw)
```
</details>

<details>
<summary>Python</summary>

```python
# p-value
p_value = 1 - f.cdf(f_stat, dfb, dfw)
```
</details>

7. Comparer avec la fonction de calcul de l'ANOVA.

<details>
<summary>R</summary>

```r
# Tableau de l'ANOVA manuel
anova_table_manual <- data.frame(
  sum_sq = c(ssb, ssw, sst),
  df = c(dfb, dfw, dfb + dfw),
  mean_sq = c(msb, msw, NA),
  F = c(f_stat, NA, NA),
  `Pr(>F)` = c(p_value, NA, NA)
)
rownames(anova_table_manual) <- c("Groupe", "Résidus", "Total")
print(anova_table_manual)

# ANOVA avec aov()
anova_model <- aov(Valeur ~ Groupe, data = result)
anova_table <- Anova(anova_model, type = 2)
print(anova_table)
```
</details>

<details>
<summary>Python</summary>

```python
# Tableau de l'ANOVA manuel
anova_table_manual = pd.DataFrame({
    'sum_sq': [ssb, ssw, sst],
    'df': [dfb, dfw, dfb + dfw],
    'mean_sq': [msb, msw, np.nan],
    'F': [f_stat, np.nan, np.nan],
    'PR(>F)': [p_value, np.nan, np.nan]
}, index=['Groupe', 'Résidus', 'Total'])

print(anova_table_manual)

model = ols('Valeur ~ Groupe', data=result).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)
```
</details>

## Exercice 2 : Cas Pratique

Pour chaque cas, poser les hypothèses, effectuer une ANOVA et interpréter le résultat.

### Lien entre Species et Sepal Length

### Lien entre Species et Sepal Width

### Lien entre Species et Petal Width

### Lien entre Species et Petal Length

## Aller plus loin :  Conditions de validité de l'ANOVA