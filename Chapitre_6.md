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
      2. [Calculer un intervalle de confiance d'une moyenne.](#calculer-un-intervalle-de-confiance-dune-moyenne)
   3. [Exercice 2 - Intervalle de confiance d'une proportion](#exercice-2---intervalle-de-confiance-dune-proportion)
      1. [Mémo](#mémo-1)
      2. [Calculer un intervalle de confiance d'une moyenne.](#calculer-un-intervalle-de-confiance-dune-moyenne-1)
   4. [Aller plus loin](#aller-plus-loin)

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

3. Calculer l'intervalle de confiance de l'estimation de la moyenne $\mu$
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

4. Calculer l'intervalle de confiance de l'estimation de la moyenne $\mu$ avec un risque  $\alpha = 0.01$
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

## Exercice 2 - Intervalle de confiance d'une proportion

:warning: Dans cet exercice, nous utilisons un échantillon de données du Titanic avec un focus sur la proportion de personne qui ne survive pas.

### Mémo
| Nom de l'indicateur | Description    | Notation | Formule                          |
|---------------------|----------------|----------|----------------------------------|
| La taille de l'échantillon | - | $n$  | -  |
| La taille de la population | Une valeur qu'on ne connaît pas la plupart du temps | $N$  | -  |
| La proportion dans l'échantillon | - | $\hat{p}$  | -  |
| La proportion dans la population | - | $P$  | -  |
| Intervalle de confiance d'une proportion | Intervalle dans lequel la proportion de la population est supposée se trouver avec un certain niveau de confiance. | -                       | $\hat{p} \pm z \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}$ où $z$ est le score z pour le niveau de confiance désiré. |
| Risque alpha                        | Probabilité de rejeter l'hypothèse nulle alors qu'elle est vraie (erreur de type I).                     | $\alpha$                | $\alpha$ est le niveau de signification choisi pour le test (généralement 0.05).               |

### Calculer un intervalle de confiance d'une moyenne. 

1. Calculer les paramètres de l'échantillon $n$ et $\hat{p}$.
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

3. Calculer l'intervalle de confiance de l'estimation de la proportion $P$
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

4. Calculer l'intervalle de confiance de l'estimation de la proportion $P$ avec un risque  $\alpha = 0.01$
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


## Aller plus loin

Bien évidemment, il existe de très nombreuses variantes pour calculer des intervalles de confiance. Selon les paramètres de l'échantillon et du contexte de l'étude, on peut être amené à utiliser d'autres formules et également d'autres lois de probabilités pour calculer le fractile $Z$. [Plus d'information ici](https://statsandr.com/blog/what-statistical-test-should-i-do/images/overview-statistical-tests-statsandr.svg).