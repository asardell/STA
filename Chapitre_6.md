# Calculer des intervalles de confiance

## Objectifs
Voici les objectifs de ce chapitre :
- [ ] Calculer un intervalle de confiance d'une moyenne
- [ ] Calculer un intervalle de confiance d'une proportion
- [ ] Comprendre la différence entre un test unilatéral et bilatéral
- [ ] Comprendre la notion de risque

1. [Calculer des intervalles de confiance](#calculer-des-intervalles-de-confiance)
   1. [Objectifs](#objectifs)
   2. [Exercice 1 - Intervalle de confiance de la moyenne](#exercice-1---intervalle-de-confiance-de-la-moyenne)
      1. [Mémo](#mémo)
      2. [Calculer un intervalle de confiance d'une moyenne.](#calculer-un-intervalle-de-confiance-dune-moyenne)

## Exercice 1 - Intervalle de confiance de la moyenne

Dans cet exercice, nous utilisons l'échantillon de données iris avec un focus sur la longueur de Petal.

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
| Intervalle de confiance d'une proportion | Intervalle dans lequel la proportion de la population est supposée se trouver avec un certain niveau de confiance. | -                       | $\hat{p} \pm z \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}$ où $z$ est le score z pour le niveau de confiance désiré. |
| Test unilatéral                     | Test statistique où l'hypothèse alternative est que le paramètre est soit plus grand soit plus petit qu'une valeur de référence. | -                       | $H_1: \mu > \mu_0$ ou $H_1: \mu < \mu_0$                                                        |
| Test bilatéral                      | Test statistique où l'hypothèse alternative est que le paramètre est différent d'une valeur de référence. | -                       | $H_1: \mu \neq \mu_0$                                                                          |
| Risque alpha                        | Probabilité de rejeter l'hypothèse nulle alors qu'elle est vraie (erreur de type I).                     | $\alpha$                | $\alpha$ est le niveau de signification choisi pour le test (généralement 0.05).               |

### Calculer un intervalle de confiance d'une moyenne. 

1. Calculer les paramètres de l'échantillon $n$ , $\bar{x}$ et $s$.
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

<details>
<summary>Excel</summary>

```
```
</details>

2. Déterminer le fractile $Z$ avec un risque $\alpha = 0.05$.
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

<details>
<summary>Excel</summary>

```
```
</details>

3. Calculer l'intervalle de confiance de la moyenne $\mu$
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

<details>
<summary>Excel</summary>

```
```
</details>
