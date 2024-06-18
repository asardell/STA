# Maitriser les statistiques descriptives

## Objectifs
Voici les objectifs de ce chapitre :
- [ ] Maitriser les indicateurs de position
- [ ] Maitriser les indicateurs de dispersion
- [ ] Représenter graphiquement une variable

Dans ce chapitre, nous allons utiliser deux jeux de données : 
- Titanic
- Iris

Les deux jeux de données sont présents par défaut dans les environnements [R](https://rdrr.io/snippets/) et [Python](https://colab.research.google.com/). Ils sont aussi accessibles dans le classeur Excel de ce repository.

:warning: les datasets peuvent avoir des différences selon le langage utilisé.

## Exercice 1 - Analyse d'une variable quantitative

### Mémo
| Nom de l'indicateur | Description    | Notation | Formule                          |
|---------------------|----------------|----------|----------------------------------|
| Moyenne             | La moyenne arithmétique est la somme des valeurs divisée par le nombre de valeurs. | $\bar{x}$ | $\bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i$ |
| Minimum             | La plus petite valeur d'un ensemble de données.            | $\min$  | $\min = x_{(1)}$               |
| Maximum             | La plus grande valeur d'un ensemble de données.            | $\max$  | $\max = x_{(n)}$               |
| Étendue             | La différence entre la plus grande et la plus petite valeur d'un ensemble de données, mesurant la dispersion totale. | $\text{étendue}$ | $\text{étendue} = \max - \min$ |
| Médiane             | La valeur qui sépare la moitié inférieure de la moitié supérieure d'un ensemble de données. | $\tilde{x}$ | - |
| Variance            | La variance mesure la dispersion des valeurs autour de la moyenne. | $s^2$ | $s^2 = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})^2$ |
| Écart-type          | L'écart-type est la racine carrée de la variance, mesurant également la dispersion des valeurs. | $s$ | $s = \sqrt{\frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})^2}$ |
| Quartiles           | Les trois valeurs qui divisent un ensemble de données trié en quatre parties égales, chaque partie représentant 25% des données. | $Q_1, Q_2, Q_3$ | Dépend de la méthode de calcul, par exemple pour $Q_1$: $Q_1 = x_{(\lceil 0.25 \cdot n \rceil)}$ |
| Déciles             | Les neuf valeurs qui divisent un ensemble de données trié en dix parties égales, chaque partie représentant 10% des données. | $D_1, D_2, ..., D_9$ | Dépend de la méthode de calcul, par exemple pour $D_1$: $D_1 = x_{(\lceil 0.1 \cdot n \rceil)}$ |
| Centiles            | Les 99 valeurs qui divisent un ensemble de données trié en cent parties égales, chaque partie représentant 1% des données. | $C_1, C_2, ..., C_{99}$ | Dépend de la méthode de calcul, par exemple pour $C_1$: $C_1 = x_{(\lceil 0.01 \cdot n \rceil)}$ |
| Histogramme         | Représentation graphique de la distribution des valeurs d'un ensemble de données sous forme de barres. | - | - |

### Exercice sur les Fonctions en R

1. Charger les données. 
<details>
<summary>R</summary>

```r
# Afficher un extrait
head(iris)
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

# Afficher un extrait
iris_df.head()
```
</details>

2. Calculer les moyennes des variables quantitatives. 
<details>
<summary>R</summary>

```r
mean(iris$Sepal.Length)
```
</details>

<details>
<summary>Python</summary>

```python
iris_df['sepal length (cm)'].mean()
```
</details>

<details>
<summary>Excel</summary>

```
MOYENNE(B:B)
```
</details>

3. Calculer les minimums des variables quantitatives. 
<details>
<summary>R</summary>

```r
min(iris$Sepal.Length)
```
</details>

<details>
<summary>Python</summary>

```python
iris_df['sepal length (cm)'].min()
```
</details>

<details>
<summary>Excel</summary>

```
MIN(B:B)
```
</details>

4. Calculer les maximums des variables quantitatives. 
<details>
<summary>R</summary>

```r
max(iris$Sepal.Length)
```
</details>

<details>
<summary>Python</summary>

```python
iris_df['sepal length (cm)'].max()
```
</details>

<details>
<summary>Excel</summary>

```
MAX(B:B)
```
</details>

5. Calculer les variances des variables quantitatives. 
<details>
<summary>R</summary>

```r
var(iris$Sepal.Length)
```
</details>

<details>
<summary>Python</summary>

```python
iris_df['sepal length (cm)'].var()
```
</details>

<details>
<summary>Excel</summary>

```
VAR(B:B)
```
</details>

6. Calculer les écart-types des variables quantitatives. 
<details>
<summary>R</summary>

```r
sd(iris$Sepal.Length)
```
</details>

<details>
<summary>Python</summary>

```python
iris_df['sepal length (cm)'].std()
```
</details>

<details>
<summary>Excel</summary>

```
ECARTYPE(B:B)
```
</details>

7. Calculer les étendus des variables quantitatives. 
<details>
<summary>R</summary>

```r
max(iris$Sepal.Length) - min(iris$Sepal.Length)
```
</details>

<details>
<summary>Python</summary>

```python
iris_df['sepal length (cm)'].max() - iris_df['sepal length (cm)'].mean()
```
</details>

<details>
<summary>Excel</summary>

```
MAX(B:B) - MIN(B:B)
```
</details>

8. Calculer les médianes des variables quantitatives. 
<details>
<summary>R</summary>

```r
median(iris$Sepal.Length)
```
</details>

<details>
<summary>Python</summary>

```python
iris_df['sepal length (cm)'].median()
```
</details>

<details>
<summary>Excel</summary>

```
MEDIANE(B:B)
```
</details>

9. Calculer les quartiles des variables quantitatives. 
<details>
<summary>R</summary>

```r
quantile(iris$Sepal.Length, probs = 0.25)
quantile(iris$Sepal.Length, probs = 0.5)
quantile(iris$Sepal.Length, probs = 0.75)
```
</details>

<details>
<summary>Python</summary>

```python
iris_df['sepal length (cm)'].quantile(0.25)
iris_df['sepal length (cm)'].quantile(0.5)
iris_df['sepal length (cm)'].quantile(0.75)
```
</details>

<details>
<summary>Excel</summary>

```
QUARTILE(B:B;1)
QUARTILE(B:B;2)
QUARTILE(B:B;3)
```
</details>

10. Calculer les déciles des variables quantitatives. 
<details>
<summary>R</summary>

```r
quantile(iris$Sepal.Length, probs = seq(from = 0.1, to = 0.9, by = 0.1))
```
</details>

<details>
<summary>Python</summary>

```python
iris_df['sepal length (cm)'].quantile([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
```
</details>

<details>
<summary>Excel</summary>

```
CENTILE(B:B;0,1)
CENTILE(B:B;0,2)
...
CENTILE(B:B;0,9)
```
</details>

11. Calculer les centiles des variables quantitatives. 
<details>
<summary>R</summary>

```r
quantile(iris$Sepal.Length, probs = seq(from = 0.01, to = 0.99, by = 0.01))
```
</details>

<details>
<summary>Python</summary>

```python
import numpy as np
iris_df['sepal length (cm)'].quantile(np.arange(0.1, 1.0, 0.01))
```
</details>

<details>
<summary>Excel</summary>

```
CENTILE(B:B;0,01)
CENTILE(B:B;0,02)
...
CENTILE(B:B;0,99)
```
</details>

12. Construire un histogramme des variables quantitatives. 
<details>
<summary>R</summary>

```r
hist(iris$Sepal.Length, main = "Histogramme Sepal.Length")
```
</details>

<details>
<summary>Python</summary>

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Choisir la colonne pour laquelle créer l'histogramme
colonne = 'sepal length (cm)'

# Créer l'histogramme avec seaborn
sns.histplot(iris_df[colonne], kde=True)

# Ajouter un titre et des labels
plt.title(f'Histogramme de {colonne}')

# Afficher l'histogramme
plt.show()
```
</details>


## Exercice 2 - Analyse d'une variable qualitatives

| Nom de l'indicateur | Description    | Notation | Formule                          |
|---------------------|----------------|----------|----------------------------------|
| Effectif            | Le nombre de fois qu'une valeur apparaît dans un ensemble de données. | $f_i$ | - |
| Effectif cumulé     | La somme des effectifs de toutes les valeurs inférieures ou égales à une valeur donnée. | $F_i$ | $F_i = \sum_{j=1}^{i} f_j$ |
| Mode                | La valeur qui apparaît le plus fréquemment dans un ensemble de données. | $\text{mode}$ | $\text{mode} = \text{valeur de } x_i \text{ telle que } f_i \text{ est maximal}$ |

1. Charger les données. 

<details>
<summary>R</summary>

```r
# Charger le jeu de données Titanic depuis titanic
library(titanic)

# Afficher un extrait
head(titanic_train)
```
</details>

<details>
<summary>Python</summary>

```python
import seaborn as sns

# Charger le jeu de données Titanic depuis seaborn
titanic = sns.load_dataset('titanic')

# Afficher un extrait
titanic.head()
```
</details>


## Liens utiles

Voici quelques liens utiles :

- [Cours sur la programmation R](https://asardell.github.io/programmation-r/)
