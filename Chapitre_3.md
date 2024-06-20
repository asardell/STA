# Lien entre deux variables qualitatives

## Objectifs
Voici les objectifs de ce chapitre :
- [ ] Représenter graphiquement deux variables quantitatives
- [ ] Interpréter des profils lignes et colonnes
- [ ] Calculer et interpréter le V de Cramer

1. [Lien entre deux variables qualitatives](#lien-entre-deux-variables-qualitatives)
   1. [Objectifs](#objectifs)
   2. [Exercice 1 - Profils lignes vs Profils colonnes entre deux variables qualitatives](#exercice-1---profils-lignes-vs-profils-colonnes-entre-deux-variables-qualitatives)
      1. [Charger les données.](#charger-les-données)
      2. [Calculer les effectifs conditionnels et marginaux.](#calculer-les-effectifs-conditionnels-et-marginaux)
      3. [Calculer les fréquences conditionnelles et marginales.](#calculer-les-fréquences-conditionnelles-et-marginales)
      4. [Calculer les profils lignes.](#calculer-les-profils-lignes)
      5. [Calculer les profils colonnes.](#calculer-les-profils-colonnes)
   3. [Exercice 2 - Représentation graphique de deux variables qualitatives](#exercice-2---représentation-graphique-de-deux-variables-qualitatives)
      1. [Diagramme en barres non empilés.](#diagramme-en-barres-non-empilés)
      2. [Diagramme en barres empilés sur les effectifs.](#diagramme-en-barres-empilés-sur-les-effectifs)
      3. [Diagramme en barres empilés sur les profils lignes.](#diagramme-en-barres-empilés-sur-les-profils-lignes)
      4. [Diagramme en barres empilés sur les profils colonnes.](#diagramme-en-barres-empilés-sur-les-profils-colonnes)
   4. [Exercice 3 - Le V de Cramer](#exercice-3---le-v-de-cramer)
      1. [Mémo](#mémo)
      2. [Calculer les effectifs observés.](#calculer-les-effectifs-observés)
      3. [Calculer les effectifs théoriques.](#calculer-les-effectifs-théoriques)
      4. [Calculer les écarts entre effectifs observés et théoriques.](#calculer-les-écarts-entre-effectifs-observés-et-théoriques)
      5. [Calculer le V de Cramer.](#calculer-le-v-de-cramer)

Dans ce chapitre, nous allons utiliser le jeu de données Iris. Il est présent par défaut dans les environnements [R](https://rdrr.io/snippets/) et [Python](https://colab.research.google.com/). Il est aussi accessible dans le classeur Excel de ce repository.

:warning: le dataset peut avoir des différences selon le langage utilisé.

## Exercice 1 - Profils lignes vs Profils colonnes entre deux variables qualitatives

:warning: On choisit d'étudier le classe des passagers et leur survie (oui/non)

### Charger les données. 
<details>
<summary>R</summary>

```r
# Charger les librairies nécessaires
library(titanic)
library(dplyr)
library(ggplot2)
library(tidyr)
library(DescTools)

# Charger le jeu de données Titanic
data("titanic_train")

# Filtrer les colonnes nécessaires
data <- titanic_train %>% select(Pclass, Survived)

```
</details>

<details>
<summary>Python</summary>

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

# Charger le jeu de données Titanic
titanic_df = sns.load_dataset('titanic')

# Filtrer les colonnes nécessaires
data = titanic_df[['class', 'survived']]
```
</details>

### Calculer les effectifs conditionnels et marginaux. 
<details>
<summary>R</summary>

```r
# Calculer les effectifs conditionnels et marginaux
contingency_table <- table(data$class, data$survived)
contingency_table_margins <- addmargins(contingency_table)
cat("Effectifs conditionnels et marginaux:\n")
print(contingency_table_margins)
```
</details>

<details>
<summary>Python</summary>

```python
# Calculer les effectifs conditionnels et marginaux
contingency_table = pd.crosstab(data['class'], data['survived'], margins=True)
print("Effectifs conditionnels et marginaux:")
print(contingency_table)
```
</details>

<details>
<summary>Excel</summary>

```
```
</details>

### Calculer les fréquences conditionnelles et marginales. 
<details>
<summary>R</summary>

```r
# Calculer les fréquences conditionnelles et marginales
frequencies <- prop.table(contingency_table_margins)
cat("\nFréquences conditionnelles et marginales:\n")
print(frequencies)
```
</details>

<details>
<summary>Python</summary>

```python
# Calculer les fréquences conditionnelles et marginales
frequencies = contingency_table / contingency_table.loc['All', 'All']
print("\nFréquences conditionnelles et marginales:")
print(frequencies)
```
</details>

<details>
<summary>Excel</summary>

```
```
</details>

### Calculer les profils lignes. 
<details>
<summary>R</summary>

```r
# Calculer les profils lignes (pourcentage lignes)
row_profiles <- prop.table(contingency_table, 1) * 100
cat("\nProfils lignes (pourcentage lignes):\n")
print(row_profiles)
```
</details>

<details>
<summary>Python</summary>

```python
# Calculer les profils lignes (pourcentage lignes)
row_profiles = pd.crosstab(data['class'], data['survived']).apply(lambda x: x/x.sum(), axis=1) * 100
print("\nProfils lignes (pourcentage lignes):")
print(row_profiles)
```
</details>

<details>
<summary>Excel</summary>

```
```
</details>

### Calculer les profils colonnes. 
<details>
<summary>R</summary>

```r
# Calculer les profils colonnes (pourcentage colonnes)
col_profiles <- prop.table(contingency_table, 2) * 100
cat("\nProfils colonnes (pourcentage colonnes):\n")
print(col_profiles)
```
</details>

<details>
<summary>Python</summary>

```python
# Calculer les profils colonnes (pourcentage colonnes)
col_profiles = pd.crosstab(data['class'], data['survived']).apply(lambda x: x/x.sum(), axis=0) * 100
print("\nProfils colonnes (pourcentage colonnes):")
print(col_profiles)
```
</details>

<details>
<summary>Excel</summary>

```
```
</details>

## Exercice 2 - Représentation graphique de deux variables qualitatives

:warning: On choisit d'étudier le classe des passagers et leur survie (oui/non)

### Diagramme en barres non empilés. 

<details>
<summary>R</summary>

```r
# Diagramme en barres non empilés
ggplot(data, aes(x = as.factor(class), fill = as.factor(survived))) +
  geom_bar(position = "dodge") +
  labs(title = "Diagramme en barres non empilés",
       x = "Classe",
       y = "Nombre de passagers",
       fill = "Survived")
```
</details>

<details>
<summary>Python</summary>

```python
# Diagramme en barres non empilés
plt.figure(figsize=(10, 6))
sns.countplot(x='class', hue='survived', data=data)
plt.title("Diagramme en barres non empilés")
plt.xlabel("Classe")
plt.ylabel("Nombre de passagers")
plt.legend(title='Survived', loc='upper right')
plt.show()
```
</details>

<details>
<summary>Excel</summary>

```
```
</details>

### Diagramme en barres empilés sur les effectifs. 

<details>
<summary>R</summary>

```r
# Diagramme en barres empilés sur les effectifs
df_contingency <- as.data.frame(contingency_table)
ggplot(df_contingency, aes(x = Var1, y = Freq, fill = Var2)) +
  geom_bar(stat = "identity") +
  labs(title = "Diagramme en barres empilés sur les effectifs",
       x = "Classe",
       y = "Nombre de passagers",
       fill = "Survived")
```
</details>

<details>
<summary>Python</summary>

```python
# Diagramme en barres empilés sur les effectifs
contingency_table_no_margins = contingency_table.drop('All', axis=0).drop('All', axis=1)
contingency_table_no_margins.plot(kind='bar', stacked=True)
plt.title("Diagramme en barres empilés sur les effectifs")
plt.xlabel("Classe")
plt.ylabel("Nombre de passagers")
plt.show()
```
</details>

<details>
<summary>Excel</summary>

```
```
</details>

### Diagramme en barres empilés sur les profils lignes. 

<details>
<summary>R</summary>

```r
# Diagramme en barres empilés sur les profils lignes
df_row_profiles <- as.data.frame(row_profiles)
ggplot(df_row_profiles, aes(x = Var1, y = Freq, fill = Var2)) +
  geom_bar(stat = "identity") +
  labs(title = "Diagramme en barres empilés sur les profils lignes",
       x = "Classe",
       y = "Pourcentage",
       fill = "Survived")
```
</details>

<details>
<summary>Python</summary>

```python
# Diagramme en barres empilés sur les profils lignes
row_profiles.plot(kind='bar', stacked=True)
plt.title("Diagramme en barres empilés sur les profils lignes")
plt.xlabel("Classe")
plt.ylabel("Pourcentage")
plt.show()
```
</details>

<details>
<summary>Excel</summary>

```
```
</details>

### Diagramme en barres empilés sur les profils colonnes. 

<details>
<summary>R</summary>

```r
# Diagramme en barres empilés sur les profils colonnes
df_col_profiles <- as.data.frame(col_profiles)
ggplot(df_col_profiles, aes(x = Var2, y = Freq, fill = Var1)) +
  geom_bar(stat = "identity") +
  labs(title = "Diagramme en barres empilés sur les profils colonnes",
       x = "Survived",
       y = "Pourcentage",
       fill = "Classe")
```
</details>

<details>
<summary>Python</summary>

```python
# Diagramme en barres empilés sur les profils colonnes
col_profiles.plot(kind='bar', stacked=True)
plt.title("Diagramme en barres empilés sur les profils colonnes")
plt.xlabel("Survived")
plt.ylabel("Pourcentage")
plt.show()
```
</details>

<details>
<summary>Excel</summary>

```
```
</details>

## Exercice 3 - Le V de Cramer

:warning: On choisit d'étudier le classe des passagers et leur survie (oui/non)

### Mémo
| Nom de l'indicateur | Description    | Notation | Formule                          |
|---------------------|----------------|----------|----------------------------------|
| Effectifs observés     | Le nombre d'observations effectivement comptées pour chaque catégorie.      | $O_i$     | - |
| Effectifs théoriques   | Le nombre d'observations attendu pour chaque catégorie selon une hypothèse. | $E_i$      | $E_i = \frac{n \cdot f_i}{N}$      |
| Chi² locaux      | Mesure de l'écart entre les effectifs observés et théoriques pour chaque catégorie. | $\chi^2_i$     | $\chi^2_i = \frac{(O_i - E_i)^2}{E_i}$        |
| Chi²  | Somme des chi² locaux, mesure globale de l'écart entre les effectifs observés et théoriques. | $\chi^2$     | $\chi^2 = \sum_{i} \frac{(O_i - E_i)^2}{E_i}$  |
| V de Cramer    | Mesure de l'association entre deux variables catégorielles.    | $V$    | $V = \sqrt{\frac{\chi^2}{n \cdot (k-1)}}$ où $\chi^2$ est la statistique chi-carré, $n$ est le nombre total d'observations, et $k$ est le nombre de catégories le plus petit entre les deux variables. |

### Calculer les effectifs observés. 

<details>
<summary>R</summary>

```r
# Calculer les effectifs observés
observed <- contingency_table
cat("\nEffectifs observés:\n")
print(observed)
```
</details>

<details>
<summary>Python</summary>

```python
# Calculer les effectifs observés
observed = pd.crosstab(data['class'], data['survived'])
print("\nEffectifs observés:")
print(observed)
```
</details>

<details>
<summary>Excel</summary>

```
```
</details>

### Calculer les effectifs théoriques. 

<details>
<summary>R</summary>

```r
# Calculer les effectifs théoriques et les écarts
chi2_test <- chisq.test(observed)
expected <- chi2_test$expected
cat("\nEffectifs théoriques:\n")
print(expected)
```
</details>

<details>
<summary>Python</summary>

```python
# Calculer les effectifs théoriques et les écarts
chi2, p, dof, expected = chi2_contingency(observed)
print("\nEffectifs théoriques:")
print(pd.DataFrame(expected, index=observed.index, columns=observed.columns))
```
</details>

<details>
<summary>Excel</summary>

```
```
</details>

### Calculer les écarts entre effectifs observés et théoriques. 

<details>
<summary>R</summary>

```r
# Calculer les écarts entre effectifs observés et théoriques
residuals <- ((observed - expected)^2) / expected
cat("\nÉcarts entre effectifs observés et théoriques:\n")
print(residuals)
```
</details>

<details>
<summary>Python</summary>

```python
# Calculer les écarts entre effectifs observés et théoriques
residuals = ((observed - expected)**2) / expected
print("\nÉcarts entre effectifs observés et théoriques:")
print(residuals)
```
</details>

<details>
<summary>Excel</summary>

```
```
</details>

### Calculer le V de Cramer. 

<details>
<summary>R</summary>

```r
# Calculer le V de Cramer
v_cramer <- CramerV(observed)
cat("\nV de Cramer:\n")
print(v_cramer)
```
</details>

<details>
<summary>Python</summary>

```python
# Calculer le V de Cramer
n = observed.sum().sum()
v_cramer = np.sqrt(chi2 / (n * (min(observed.shape) - 1)))
print("\nV de Cramer:")
print(v_cramer)
```
</details>

<details>
<summary>Excel</summary>

```
```
</details>