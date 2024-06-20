# Maitriser les statistiques descriptives

## Objectifs
Voici les objectifs de ce chapitre :
- [ ] Maitriser les indicateurs de position
- [ ] Maitriser les indicateurs de dispersion
- [ ] Représenter graphiquement une variable

1. [Maitriser les statistiques descriptives](#maitriser-les-statistiques-descriptives)
   1. [Objectifs](#objectifs)
   2. [Exercice 1 - Analyse d'une variable quantitative](#exercice-1---analyse-dune-variable-quantitative)
      1. [Mémo](#mémo)
      2. [Charger les données.](#charger-les-données)
      3. [Calculer les moyennes des variables quantitatives.](#calculer-les-moyennes-des-variables-quantitatives)
      4. [Calculer les minimums des variables quantitatives.](#calculer-les-minimums-des-variables-quantitatives)
      5. [Calculer les maximums des variables quantitatives.](#calculer-les-maximums-des-variables-quantitatives)
      6. [Calculer les variances des variables quantitatives.](#calculer-les-variances-des-variables-quantitatives)
      7. [Calculer les écart-types des variables quantitatives.](#calculer-les-écart-types-des-variables-quantitatives)
      8. [Calculer les étendus des variables quantitatives.](#calculer-les-étendus-des-variables-quantitatives)
      9. [Calculer les médianes des variables quantitatives.](#calculer-les-médianes-des-variables-quantitatives)
      10. [Calculer les quartiles des variables quantitatives.](#calculer-les-quartiles-des-variables-quantitatives)
      11. [Calculer l'écart interquartile des variables quantitatives.](#calculer-lécart-interquartile-des-variables-quantitatives)
      12. [Calculer les déciles des variables quantitatives.](#calculer-les-déciles-des-variables-quantitatives)
      13. [Calculer les centiles des variables quantitatives.](#calculer-les-centiles-des-variables-quantitatives)
      14. [Construire un histogramme des variables quantitatives.](#construire-un-histogramme-des-variables-quantitatives)
      15. [Construire un boxplot des variables quantitatives.](#construire-un-boxplot-des-variables-quantitatives)
   3. [Exercice 2 - Analyse d'une variable qualitative](#exercice-2---analyse-dune-variable-qualitative)
      1. [Charger les données.](#charger-les-données-1)
      2. [Calculer les effectifs des classes des passagers.](#calculer-les-effectifs-des-classes-des-passagers)
      3. [Calculer les fréquences des classes des passagers.](#calculer-les-fréquences-des-classes-des-passagers)
      4. [Calculer les effectifs cumulés des classes des passagers.](#calculer-les-effectifs-cumulés-des-classes-des-passagers)
      5. [Construire un diagramme en barres.](#construire-un-diagramme-en-barres)
      6. [Construire un diagramme circulaire.](#construire-un-diagramme-circulaire)

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
| Variance            | La variance mesure la dispersion des valeurs autour de la moyenne. | $s^2$ | $s^2 = \frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{x})^2$ |
| Écart-type          | L'écart-type est la racine carrée de la variance, mesurant également la dispersion des valeurs. | $s$ | $s = \sqrt{s^2}$ |
| Quartiles           | Les trois valeurs qui divisent un ensemble de données trié en quatre parties égales, chaque partie représentant 25% des données. | $Q_1, Q_2, Q_3$ | Dépend de la méthode de calcul, par exemple pour $Q_1$: $Q_1 = x_{(\lceil 0.25 \cdot n \rceil)}$ |
| Écart interquartile | La différence entre le troisième et le premier quartile, mesurant la dispersion au milieu de la distribution. | $IQR$ | $IQR = Q_3 - Q_1$ |
| Déciles             | Les neuf valeurs qui divisent un ensemble de données trié en dix parties égales, chaque partie représentant 10% des données. | $D_1, D_2, ..., D_9$ | Dépend de la méthode de calcul, par exemple pour $D_1$: $D_1 = x_{(\lceil 0.1 \cdot n \rceil)}$ |
| Centiles            | Les 99 valeurs qui divisent un ensemble de données trié en cent parties égales, chaque partie représentant 1% des données. | $C_1, C_2, ..., C_{99}$ | Dépend de la méthode de calcul, par exemple pour $C_1$: $C_1 = x_{(\lceil 0.01 \cdot n \rceil)}$ |
| Histogramme         | Représentation graphique de la distribution des valeurs d'un ensemble de données sous forme de barres. | - | - |
| Boxplot         | Représentation graphique de la distribution des valeurs d'un ensemble de données sous forme de boîte. | - | - |

### Charger les données. 
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

### Calculer les moyennes des variables quantitatives. 
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

### Calculer les minimums des variables quantitatives. 
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

### Calculer les maximums des variables quantitatives. 
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

### Calculer les variances des variables quantitatives. 
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

### Calculer les écart-types des variables quantitatives. 
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

### Calculer les étendus des variables quantitatives. 
<details>
<summary>R</summary>

```r
max(iris$Sepal.Length) - min(iris$Sepal.Length)
```
</details>

<details>
<summary>Python</summary>

```python
iris_df['sepal length (cm)'].max() - iris_df['sepal length (cm)'].min()
```
</details>

<details>
<summary>Excel</summary>

```
MAX(B:B) - MIN(B:B)
```
</details>

### Calculer les médianes des variables quantitatives. 
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

### Calculer les quartiles des variables quantitatives. 
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

### Calculer l'écart interquartile des variables quantitatives. 
<details>
<summary>R</summary>

```r
q1 = quantile(iris$Sepal.Length, probs = 0.25)
q3 = quantile(iris$Sepal.Length, probs = 0.75)
q3 - q1
```
</details>

<details>
<summary>Python</summary>

```python
q1 = iris_df['sepal length (cm)'].quantile(0.25)
q3 = iris_df['sepal length (cm)'].quantile(0.75)
q3 - q1
```
</details>

<details>
<summary>Excel</summary>

```
QUARTILE(B:B;3) - QUARTILE(B:B;1)
```
</details>

### Calculer les déciles des variables quantitatives. 
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

### Calculer les centiles des variables quantitatives. 
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

### Construire un histogramme des variables quantitatives. 
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


### Construire un boxplot des variables quantitatives. 

Plus d'info sur l'interprétation d'un boxplot avec deux approches : 
- [Avec outliers](https://cdn1.byjus.com/wp-content/uploads/2020/10/Box-Plot-and-Whisker-Plot-1.png)
- [Sans outliers](https://media.geeksforgeeks.org/wp-content/uploads/20201127012952/boxplot-660x233.png)

<details>
<summary>R</summary>

```r
boxplot(iris$Sepal.Length, main = "Boxplot Sepal.Length")
```
</details>

<details>
<summary>Python</summary>

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Charger le jeu de données Iris depuis seaborn
iris = sns.load_dataset('iris')

# Créer un boxplot de 'sepal_length'
plt.figure(figsize=(10, 6))
sns.boxplot(x=iris['sepal_length'])
plt.title('Boxplot de la longueur des sépales (cm)')

# Afficher le boxplot
plt.show()
```
</details>


## Exercice 2 - Analyse d'une variable qualitative

| Nom de l'indicateur | Description    | Notation | Formule                          |
|---------------------|----------------|----------|----------------------------------|
| Effectif            | Le nombre de fois qu'une valeur apparaît dans un ensemble de données. | $f_i$ | - |
| Effectif cumulé     | La somme des effectifs de toutes les valeurs inférieures ou égales à une valeur donnée. | $F_i$ | $F_i = \sum_{j=1}^{i} f_j$ |
| Mode                | La valeur qui apparaît le plus fréquemment dans un ensemble de données. | $\text{mode}$ | $\text{mode} = \text{valeur de } x_i \text{ telle que } f_i \text{ est maximal}$ |
| Diagramme en barres | Représentation graphique de la fréquence des différentes valeurs d'un ensemble de données sous forme de barres. | - | - |
| Diagramme circulaire | Représentation graphique des proportions des différentes valeurs d'un ensemble de données sous forme de secteurs d'un disque. | - | - |

### Charger les données. 
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

### Calculer les effectifs des classes des passagers. 
<details>
<summary>R</summary>

```r
table(titanic_train$Pclass)
```
</details>

<details>
<summary>Python</summary>

```python
titanic['pclass'].value_counts()
```
</details>

<details>
<summary>Excel</summary>

```
Tableau croisé dynamique
```
</details>

### Calculer les fréquences des classes des passagers. 
<details>
<summary>R</summary>

```r
effectif = table(titanic_train$Pclass)
prop.table(effectif)
```
</details>

<details>
<summary>Python</summary>

```python
titanic['pclass'].value_counts(normalize=True)
```
</details>

<details>
<summary>Excel</summary>

```
Tableau croisé dynamique
```
</details>

### Calculer les effectifs cumulés des classes des passagers. 
<details>
<summary>R</summary>

```r
# Compter les effectifs des modalités de la variable Pclass
effectifs_Pclass <- table(titanic_train$Pclass)

# Calculer les effectifs cumulés
effectifs_cumules_Pclass <- cumsum(effectifs_Pclass)

# Afficher les effectifs cumulés
print("Effectifs cumulés des différentes classes de passagers :")
print(effectifs_cumules_Pclass)
```
</details>

<details>
<summary>Python</summary>

```python
# Calculer les effectifs des différentes classes de passagers
effectifs_pclass = titanic['pclass'].value_counts().sort_index()

# Calculer les effectifs cumulés
effectifs_cumules = effectifs_pclass.cumsum()

# Affichage des effectifs et des effectifs cumulés
print("Effectifs cumulés des différentes classes de passagers :")
print(effectifs_cumules)
```
</details>

<details>
<summary>Excel</summary>

```
Tableau croisé dynamique
```
</details>

### Construire un diagramme en barres. 
<details>
<summary>R</summary>

```r
# Compter les effectifs des modalités de la variable Pclass
effectifs_Pclass <- table(titanic_train$Pclass)
barplot(effectifs_Pclass, main = "Diagramme Pclass")
```
</details>

<details>
<summary>Python</summary>

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Créer le compte des valeurs de 'pclass'
effectifs_pclass = titanic['pclass'].value_counts().sort_index()

# Utiliser seaborn pour créer un diagramme en barres
plt.figure(figsize=(8, 6))
sns.barplot(x=effectifs_pclass.index, y=effectifs_pclass.values, palette='viridis')

# Ajouter des titres et des labels
plt.title('Effectifs des classes de passagers')
plt.xticks(ticks=[0, 1, 2], labels=['1ère classe', '2ème classe', '3ème classe'])

# Afficher le diagramme
plt.show()
```
</details>

### Construire un diagramme circulaire. 
<details>
<summary>R</summary>

```r
# Compter les effectifs des modalités de la variable Pclass
effectifs_Pclass <- table(titanic_train$Pclass)
pie(effectifs_Pclass, main = "Diagramme Pclass")
```
</details>

<details>
<summary>Python</summary>

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Créer le compte des valeurs de 'pclass'
effectifs_pclass = titanic['pclass'].value_counts().sort_index()

# Définir les labels et les effectifs pour le diagramme
labels = ['1ère classe', '2ème classe', '3ème classe']
sizes = effectifs_pclass.values

# Créer le diagramme circulaire
plt.figure(figsize=(8, 6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('viridis', 3))
plt.title('Répartition des classes de passagers dans le Titanic')

# Afficher le diagramme
plt.axis('equal')  # Assure que le diagramme est circulaire
plt.show()
```
</details>

