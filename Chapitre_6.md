# Calculer des intervalles de confiance

## Objectifs
Voici les objectifs de ce chapitre :
- [ ] Calculer un intervalle de confiance d'une moyenne
- [ ] Calculer un intervalle de confiance d'une proportion
- [ ] Comprendre la notion de risque

1. [Calculer des intervalles de confiance](#calculer-des-intervalles-de-confiance)
   1. [Objectifs](#objectifs)
   2. [Exercice 1 - Intervalle de confiance de la moyenne](#exercice-1---intervalle-de-confiance-de-la-moyenne)
      1. [Mémo](#mémo)
      2. [Charger les données.](#charger-les-données)
      3. [Calculer un intervalle de confiance d'une moyenne.](#calculer-un-intervalle-de-confiance-dune-moyenne)
   3. [Exercice 2 - Intervalle de confiance d'une proportion](#exercice-2---intervalle-de-confiance-dune-proportion)
      1. [Mémo](#mémo-1)
      2. [Charger les données.](#charger-les-données-1)
      3. [Calculer un intervalle de confiance d'une proportion.](#calculer-un-intervalle-de-confiance-dune-proportion)
   4. [Aller plus loin.](#aller-plus-loin)

## Exercice 1 - Intervalle de confiance de la moyenne

:warning: Dans cet exercice, nous utilisons l'échantillon de données iris avec un focus sur la longueur de Petal.

### Mémo
| Nom de l'indicateur | Description    | Notation | Formule                          |
|---------------------|----------------|----------|----------------------------------|
| La taille de l'échantillon | - | $n$  | -  |
| Moyenne de l'échantillon | - | $\bar{x}$  | -  |
| Ecart-type de l'échantillon | - | $s$  | -  |
| La taille de la population | Une valeur qu'on ne connaît pas la plupart du temps | $N$  | -  |
| Moyenne de la population | Une valeur qu'on cherche à estimer | $\mu$  | -  |
| Ecart-type de la population | Une valeur qu'on cherche à estimer | $\sigma$  | -  |
| Intervalle de confiance d'une moyenne | Intervalle dans lequel la moyenne de la population est supposée se trouver avec un certain niveau de confiance. | -                       | $\bar{x} \pm z \frac{s}{\sqrt{n}}$ où $z$ est le score z pour le niveau de confiance désiré.    |
| Risque alpha                        | Probabilité de rejeter l'hypothèse nulle alors qu'elle est vraie (erreur de type I).                     | $\alpha$                | $\alpha$ est le niveau de signification choisi pour le test (généralement 0.05).               |

### Charger les données. 
<details>
<summary>R</summary>

```r
# Charger les données Iris
data(iris)
petal_length <- iris$Petal.Length
```
</details>

<details>
<summary>Python</summary>

```python
import numpy as np
from scipy import stats
from sklearn.datasets import load_iris

# Charger les données Iris
iris = load_iris()
petal_length = iris.data[:, 2]  # Longueur de Petal
```
</details>

### Calculer un intervalle de confiance d'une moyenne. 

1. Calculer les paramètres de l'échantillon $n$ , $\bar{x}$ et $s$.
<details>
<summary>R</summary>

```r
# Calculer les paramètres de l'échantillon n, x̄ et s
n <- length(petal_length)
mean_petal_length <- mean(petal_length)
std_dev_petal_length <- sd(petal_length)

cat("Taille de l'échantillon (n) :", n, "\n")
cat("Moyenne (x̄) :", mean_petal_length, "\n")
cat("Écart-type (s) :", std_dev_petal_length, "\n")
```
</details>

<details>
<summary>Python</summary>

```python
# Calculer les paramètres de l'échantillon n, x̄ et s
n = len(petal_length)
mean_petal_length = np.mean(petal_length)
std_dev_petal_length = np.std(petal_length, ddof=1)

print(f"Taille de l'échantillon (n) : {n}")
print(f"Moyenne (x̄) : {mean_petal_length}")
print(f"Écart-type (s) : {std_dev_petal_length}")
```
</details>

<details>
<summary>Excel</summary>

```
```
</details>

2. Déterminer le fractile $Z$ avec un risque $\alpha = 0.05$.
<details>
<summary>R</summary>

```r
# Déterminer le fractile Z avec un risque α = 0.05
alpha_05 <- 0.05
z_05 <- qnorm(1 - alpha_05 / 2)
cat("Fractile Z pour un risque α = 0.05 :", z_05, "\n")
```
</details>

<details>
<summary>Python</summary>

```python
# Déterminer le fractile Z avec un risque α = 0.05
alpha_05 = 0.05
z_05 = stats.norm.ppf(1 - alpha_05 / 2)
print(f"Fractile Z pour un risque α = 0.05 : {z_05}")
```
</details>

<details>
<summary>Excel</summary>

```
```
</details>

3. Calculer l'intervalle de confiance de l'estimation de la moyenne $\mu$
<details>
<summary>R</summary>

```r
# Calculer l'intervalle de confiance de l'estimation de la moyenne μ avec α = 0.05
margin_of_error_05 <- z_05 * (std_dev_petal_length / sqrt(n))
confidence_interval_05 <- c(mean_petal_length - margin_of_error_05, mean_petal_length + margin_of_error_05)
cat("Intervalle de confiance pour μ avec α = 0.05 :", confidence_interval_05, "\n")
```
</details>

<details>
<summary>Python</summary>

```python
# Calculer l'intervalle de confiance de l'estimation de la moyenne μ avec α = 0.05
margin_of_error_05 = z_05 * (std_dev_petal_length / np.sqrt(n))
confidence_interval_05 = (mean_petal_length - margin_of_error_05, mean_petal_length + margin_of_error_05)
print(f"Intervalle de confiance pour μ avec α = 0.05 : {confidence_interval_05}")
```
</details>

<details>
<summary>Excel</summary>

```
```
</details>

4. Calculer l'intervalle de confiance de l'estimation de la moyenne $\mu$ avec un risque  $\alpha = 0.01$
<details>
<summary>R</summary>

```r
# Calculer l'intervalle de confiance de l'estimation de la moyenne μ avec un risque α = 0.01
alpha_01 <- 0.01
z_01 <- qnorm(1 - alpha_01 / 2)
cat("Fractile Z pour un risque α = 0.01 :", z_01, "\n")

margin_of_error_01 <- z_01 * (std_dev_petal_length / sqrt(n))
confidence_interval_01 <- c(mean_petal_length - margin_of_error_01, mean_petal_length + margin_of_error_01)
cat("Intervalle de confiance pour μ avec α = 0.01 :", confidence_interval_01, "\n")
```
</details>

<details>
<summary>Python</summary>

```python
# Calculer l'intervalle de confiance de l'estimation de la moyenne μ avec un risque α = 0.01
alpha_01 = 0.01
z_01 = stats.norm.ppf(1 - alpha_01 / 2)
print(f"Fractile Z pour un risque α = 0.01 : {z_01}")

margin_of_error_01 = z_01 * (std_dev_petal_length / np.sqrt(n))
confidence_interval_01 = (mean_petal_length - margin_of_error_01, mean_petal_length + margin_of_error_01)
print(f"Intervalle de confiance pour μ avec α = 0.01 : {confidence_interval_01}")
```
</details>

<details>
<summary>Excel</summary>

```
```
</details>

## Exercice 2 - Intervalle de confiance d'une proportion

:warning: Dans cet exercice, nous utilisons un échantillon de données du Titanic avec un focus sur la proportion de personne qui ne survit pas.

### Mémo
| Nom de l'indicateur | Description    | Notation | Formule                          |
|---------------------|----------------|----------|----------------------------------|
| La taille de l'échantillon | - | $n$  | -  |
| La taille de la population | Une valeur qu'on ne connaît pas la plupart du temps | $N$  | -  |
| La proportion dans l'échantillon | - | $\hat{p}$  | -  |
| La proportion dans la population | - | $P$  | -  |
| Intervalle de confiance d'une proportion | Intervalle dans lequel la proportion de la population est supposée se trouver avec un certain niveau de confiance. | -                       | $\hat{p} \pm z \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}$ où $z$ est le score z pour le niveau de confiance désiré. |
| Risque alpha                        | Probabilité de rejeter l'hypothèse nulle alors qu'elle est vraie (erreur de type I).                     | $\alpha$                | $\alpha$ est le niveau de signification choisi pour le test (généralement 0.05).               |

### Charger les données. 
<details>
<summary>R</summary>

```r
# Charger les données Titanic
library(titanic)
data("titanic_train")
titanic <- titanic_train
```
</details>

<details>
<summary>Python</summary>

```python
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns

# Charger les données Titanic
titanic = sns.load_dataset('titanic')
```
</details>

### Calculer un intervalle de confiance d'une proportion. 

1. Calculer les paramètres de l'échantillon $n$ et $\hat{p}$.
<details>
<summary>R</summary>

```r
# Calculer les paramètres de l'échantillon n et p̂
n <- nrow(titanic)
p_hat <- mean(titanic$Survived, na.rm = TRUE)

cat("Taille de l'échantillon (n) :", n, "\n")
cat("Proportion de survivants (p̂) :", p_hat, "\n")
```
</details>

<details>
<summary>Python</summary>

```python
# Calculer les paramètres de l'échantillon n et p̂
n = len(titanic)
p_hat = titanic['survived'].mean()

print(f"Taille de l'échantillon (n) : {n}")
print(f"Proportion de survivants (p̂) : {p_hat}")
```
</details>

<details>
<summary>Excel</summary>

```
```
</details>

2. Déterminer le fractile $Z$ avec un risque $\alpha = 0.05$.

:bulb: Comment répartir le risque dans un cas de [test bilatéral](https://slideplayer.fr/slide/13724665/85/images/7/Test+bilat%C3%A9ral+zone+de+rejet+zone+d%E2%80%99acceptation.jpg) ? 

<details>
<summary>R</summary>

```r
# Déterminer le fractile Z avec un risque α = 0.05
alpha_05 <- 0.05
z_05 <- qnorm(1 - alpha_05 / 2)
cat("Fractile Z pour un risque α = 0.05 :", z_05, "\n")
```
</details>

<details>
<summary>Python</summary>

```python
# Déterminer le fractile Z avec un risque α = 0.05
alpha_05 = 0.05
z_05 = stats.norm.ppf(1 - alpha_05 / 2)
print(f"Fractile Z pour un risque α = 0.05 : {z_05}")
```
</details>

<details>
<summary>Excel</summary>

```
```
</details>

3. Calculer l'intervalle de confiance de l'estimation de la proportion $P$
<details>
<summary>R</summary>

```r
# Calculer l'intervalle de confiance de l'estimation de la proportion P avec α = 0.05
margin_of_error_05 <- z_05 * sqrt(p_hat * (1 - p_hat) / n)
confidence_interval_05 <- c(p_hat - margin_of_error_05, p_hat + margin_of_error_05)
cat("Intervalle de confiance pour P avec α = 0.05 :", confidence_interval_05, "\n")

```
</details>

<details>
<summary>Python</summary>

```python
# Calculer l'intervalle de confiance de l'estimation de la proportion P avec α = 0.05
margin_of_error_05 = z_05 * np.sqrt(p_hat * (1 - p_hat) / n)
confidence_interval_05 = (p_hat - margin_of_error_05, p_hat + margin_of_error_05)
print(f"Intervalle de confiance pour P avec α = 0.05 : {confidence_interval_05}")
```
</details>

<details>
<summary>Excel</summary>

```
```
</details>

4. Calculer l'intervalle de confiance de l'estimation de la proportion $P$ avec un risque  $\alpha = 0.01$
<details>
<summary>R</summary>

```r
# Calculer l'intervalle de confiance de l'estimation de la proportion P avec un risque α = 0.01
alpha_01 <- 0.01
z_01 <- qnorm(1 - alpha_01 / 2)
cat("Fractile Z pour un risque α = 0.01 :", z_01, "\n")

margin_of_error_01 <- z_01 * sqrt(p_hat * (1 - p_hat) / n)
confidence_interval_01 <- c(p_hat - margin_of_error_01, p_hat + margin_of_error_01)
cat("Intervalle de confiance pour P avec α = 0.01 :", confidence_interval_01, "\n")
```
</details>

<details>
<summary>Python</summary>

```python
# Calculer l'intervalle de confiance de l'estimation de la proportion P avec un risque α = 0.01
alpha_01 = 0.01
z_01 = stats.norm.ppf(1 - alpha_01 / 2)
print(f"Fractile Z pour un risque α = 0.01 : {z_01}")

margin_of_error_01 = z_01 * np.sqrt(p_hat * (1 - p_hat) / n)
confidence_interval_01 = (p_hat - margin_of_error_01, p_hat + margin_of_error_01)
print(f"Intervalle de confiance pour P avec α = 0.01 : {confidence_interval_01}")
```
</details>

<details>
<summary>Excel</summary>

```
```
</details>


## Aller plus loin.

Bien évidemment, il existe de très nombreuses variantes pour calculer des intervalles de confiance. Selon les paramètres de l'échantillon et du contexte de l'étude, on peut être amené à utiliser d'autres formules et également d'autres lois de probabilités pour calculer le fractile $Z$. [Plus d'information ici](https://statsandr.com/blog/what-statistical-test-should-i-do/images/overview-statistical-tests-statsandr.svg).