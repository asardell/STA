# Le test du Chi²

## Objectifs
Voici les objectifs de ce chapitre :
- [ ] Calculer et interpréter un test du Chi²
- [ ] Connaître les limites du test du Chi²
- [ ] Analyser la p-value
- [ ] Comprendre l'intérêt des hypothèses de départ et alternative

1. [Le test du Chi²](#le-test-du-chi)
   1. [Objectifs](#objectifs)
   2. [Exercice 1 : Cas théorique](#exercice-1--cas-théorique)
      1. [Mémo](#mémo)
   3. [Exercice 2 : Cas Pratique](#exercice-2--cas-pratique)
      1. [Lien entre la classe des passagers et leur survie](#lien-entre-la-classe-des-passagers-et-leur-survie)
      2. [Lien entre la classe des passagers et le genre](#lien-entre-la-classe-des-passagers-et-le-genre)
      3. [Lien entre le genre des passagers et leur survie.](#lien-entre-le-genre-des-passagers-et-leur-survie)

## Exercice 1 : Cas théorique

### Mémo
| Nom de l'indicateur | Description    | Notation | Formule                          |
|---------------------|----------------|----------|----------------------------------|
| **Test du Khi-2** | Mesure de l'association entre deux variables catégorielles en comparant les fréquences observées avec les fréquences attendues. | $\chi^2$ | $\chi^2 = \sum \frac{(O_i - E_i)^2}{E_i}$ où $O_i$ est la fréquence observée et $E_i$ est la fréquence attendue. |
| **La taille de l'échantillon** | - | $n$ | - |
| **Degrés de liberté** | Nombre de valeurs libres de varier dans le calcul d'une statistique | $df$ | $(r - 1)(c - 1)$ où $r$ est le nombre de lignes et $c$ est le nombre de colonnes. |
| **Statistique $\chi^2$** | Valeur calculée pour tester l'hypothèse nulle dans le test du Khi-2 | $\chi^2$ | $\chi^2 = \sum \frac{(O_i - E_i)^2}{E_i}$ |
| **P-value** | Probabilité d'obtenir une statistique $\chi^2$ au moins aussi extrême que celle observée, sous l'hypothèse nulle | - | Déterminée à partir de la distribution $\chi^2$ avec $(r - 1)(c - 1)$ degrés de liberté |

1. Simuler un tableau de contingence.

<details>
<summary>R</summary>

```r
# Charger les bibliothèques nécessaires
library(stats)

# Définir la matrice des observations
obs <- matrix(c(693, 886, 534, 153, 597, 696, 448, 95), nrow = 2, byrow = TRUE)
```
</details>

<details>
<summary>Python</summary>

```python
import numpy as np
obs = np.array([[693,886,534,153], [597,696,448,95]])
```
</details>


2. Poser $H_0$ et $H_1$.
- $H_0$ : Les variables sont indépendantes
- $H_1$ : Les variables ne sont pas indépendantes

3. Effectuer un test du Chi².
<details>
<summary>R</summary>

```r
# Calculer le test du Khi-2
chi2_test <- chisq.test(obs)

# Afficher les résultats
cat('Khi2  :', chi2_test$statistic, '\n')
cat('p_value  :', chi2_test$p.value, '\n')
cat('effectif_theorique  :\n')
print(chi2_test$expected)
cat('ddl  :', chi2_test$parameter, '\n')
```
</details>

<details>
<summary>Python</summary>

```python
from scipy.stats import chi2_contingency
Khi2_obs, p_value, ddl, effectif_theorique = chi2_contingency(obs)
print(f'Khi2  : {Khi2_obs}')
print(f'p_value  : {p_value}')
print(f'effectif_theorique  : {effectif_theorique}')
print(f'ddl  : {ddl}')
```
</details>

4. Calculer manuellement la p-value.

:bulb: On cherche quelle est la probabilité critique pour laquelle $Khi2_obs < Khi2_max$ dans [la table du Khi²](https://i0.wp.com/statisticsbyjim.com/wp-content/uploads/2022/01/chi-square_table.png?resize=625%2C800&ssl=1).

<details>
<summary>R</summary>

```r
# Calculer les valeurs critiques du Khi-2 pour différentes combinaisons de ddl et de niveaux de confiance
J <- 1:4
I <- seq(0.05, 0.15, 0.005)

# Initialiser une matrice pour stocker les valeurs critiques
a <- matrix(NA, nrow = length(J), ncol = length(I))

# Remplir la matrice avec les valeurs critiques
for (i in seq_along(I)) {
  for (j in seq_along(J)) {
    a[j, i] <- qchisq(1 - I[i], df = J[j])
  }
}

# Convertir la matrice en data frame avec des noms de colonnes et de lignes
df_chi2 <- round(as.data.frame(a), 5)
colnames(df_chi2) <- I
rownames(df_chi2) <- J

# Afficher le data frame
print(df_chi2)

```
</details>

<details>
<summary>Python</summary>

```python
from scipy.stats import chi2
J = df = np.arange(1,5,1)
I = np.arange(0.05,0.15,0.005)

a = np.empty((len(J),len(I)))
a[:] = np.nan

for i in range(0,len(I)):
    for j in range(0,len(J)):
        a[j,i] = chi2.isf(I[i], J[j])
        
df_chi2 = round(pd.DataFrame(a, columns=I, index = J),5)
df_chi2
```
</details>

## Exercice 2 : Cas Pratique

### Lien entre la classe des passagers et leur survie

<details>
<summary>Python</summary>

```python
import seaborn as sns

# Charger le jeu de données Titanic depuis seaborn
titanic = sns.load_dataset('titanic')

# Afficher un extrait
titanic.head()

import numpy as np
from scipy.stats import chi2_contingency
obs = pd.crosstab(titanic.pclass, titanic.survived)

Khi2_obs, p_value, ddl, effectif_theorique = chi2_contingency(obs)

print(f'Khi2  : {Khi2_obs}')
print(f'p_value  : {p_value}')
print(f'effectif_theorique  : {effectif_theorique}')
print(f'ddl  : {ddl}')
```
</details>

### Lien entre la classe des passagers et le genre

<details>
<summary>Python</summary>

```python
import numpy as np
from scipy.stats import chi2_contingency
obs = pd.crosstab(titanic.pclass, titanic.sex)

Khi2_obs, p_value, ddl, effectif_theorique = chi2_contingency(obs)

print(f'Khi2  : {Khi2_obs}')
print(f'p_value  : {p_value}')
print(f'effectif_theorique  : {effectif_theorique}')
print(f'ddl  : {ddl}')
```
</details>

### Lien entre le genre des passagers et leur survie.

<details>
<summary>Python</summary>

```python
import numpy as np
from scipy.stats import chi2_contingency
obs = pd.crosstab(titanic.sex, titanic.survived)

Khi2_obs, p_value, ddl, effectif_theorique = chi2_contingency(obs)

print(f'Khi2  : {Khi2_obs}')
print(f'p_value  : {p_value}')
print(f'effectif_theorique  : {effectif_theorique}')
print(f'ddl  : {ddl}')
```
</details>