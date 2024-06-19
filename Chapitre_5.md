# Découvrir la loi uniforme et la loi normale

## Objectifs
Voici les objectifs de ce chapitre :
- [ ] Simuler une loi uniforme
- [ ] Simuler une loi normale
- [ ] Découvrir la loi normale centrée réduite
- [ ] Lire dans une table de loi normale

1. [Découvrir la loi uniforme et la loi normale](#découvrir-la-loi-uniforme-et-la-loi-normale)
   1. [Objectifs](#objectifs)
   2. [Exercice 1 - La loi uniforme](#exercice-1---la-loi-uniforme)
      1. [Mémo](#mémo)
      2. [Echantillon de taille 30 suivant une loi uniforme de paramètre $U(0, 1)$.](#echantillon-de-taille-30-suivant-une-loi-uniforme-de-paramètre-u0-1)
      3. [Echantillon de taille 2000 suivant une loi uniforme de paramètre $U(0, 1)$.](#echantillon-de-taille-2000-suivant-une-loi-uniforme-de-paramètre-u0-1)
   3. [Exercice 2 - La loi normale](#exercice-2---la-loi-normale)
      1. [Mémo](#mémo-1)
      2. [Echantillon de taille 30 suivant une loi normale de paramètre $N(\\mu = 4, \\sigma = 2)$.](#echantillon-de-taille-30-suivant-une-loi-normale-de-paramètre-nmu--4-sigma--2)
      3. [Echantillon de taille 2000 suivant une loi normale de paramètre $N(\\mu = 4, \\sigma = 2)$.](#echantillon-de-taille-2000-suivant-une-loi-normale-de-paramètre-nmu--4-sigma--2)
      4. [Echantillon de taille 2000 suivant une loi normale de paramètre $N(\\mu = 4, \\sigma = 6)$.](#echantillon-de-taille-2000-suivant-une-loi-normale-de-paramètre-nmu--4-sigma--6)
      5. [Echantillon de taille 2000 suivant une loi normale de paramètre $N(\\mu = 0, \\sigma = 1)$.](#echantillon-de-taille-2000-suivant-une-loi-normale-de-paramètre-nmu--0-sigma--1)
   4. [Exercice 3 - Lecture de la table de la loi normale](#exercice-3---lecture-de-la-table-de-la-loi-normale)
      1. [Mémo](#mémo-2)
      2. [Comprendre la table de la loi normale centrée réduite.](#comprendre-la-table-de-la-loi-normale-centrée-réduite)
      3. [Calculer la probabilité cumulative supérieure.](#calculer-la-probabilité-cumulative-supérieure)
   5. [Aller plus loin](#aller-plus-loin)

## Exercice 1 - La loi uniforme

### Mémo
| Nom de l'indicateur | Description    | Notation | Formule                          |
|---------------------|----------------|----------|----------------------------------|
| Loi uniforme               | Distribution où toutes les valeurs dans un intervalle sont également probables. | $U(a, b)$        | $f(x) = \frac{1}{b-a}$ pour $a \leq x \leq b$          |

### Echantillon de taille 30 suivant une loi uniforme de paramètre $U(0, 1)$. 

1. Simuler un échantillon de taille 30 suivant une loi uniforme de paramètre $U(0, 1)$
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

2. Calculer la moyenne, le minimum, le maximum, la médiane et l'écart-type
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

3. Représenter graphiquement la distribution dans un histogramme
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

### Echantillon de taille 2000 suivant une loi uniforme de paramètre $U(0, 1)$. 

1. Simuler un échantillon de taille 2000 suivant une loi uniforme de paramètre $U(0, 1)$
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

2. Calculer la moyenne, le minimum, le maximum, la médiane et l'écart-type
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

3. Représenter graphiquement la distribution dans un histogramme
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

## Exercice 2 - La loi normale

### Mémo
| Nom de l'indicateur | Description    | Notation | Formule                          |
|---------------------|----------------|----------|----------------------------------|
| Loi normale                | Distribution de probabilité continue symétrique, en forme de cloche, décrite par la moyenne et l'écart-type. | $N(\mu, \sigma)$ | $f(x) = \frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$   |
| Fonction de répartition    | Fonction qui donne la probabilité que la variable aléatoire soit inférieure ou égale à une valeur donnée. | $F(x)$           | $F(x) = P(X \leq x)$      |
| Fonction densité           | Fonction qui décrit la probabilité de la variable aléatoire continue.        | $f(x)$           | $f(x) = \frac{dF(x)}{dx}$ pour une variable continue          |

### Echantillon de taille 30 suivant une loi normale de paramètre $N(\mu = 4, \sigma = 2)$. 

1. Simuler un échantillon de taille 30 suivant une loi uniforme de paramètre $N(\mu = 4, \sigma = 2)$
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

2. Calculer la moyenne, le minimum, le maximum, la médiane et l'écart-type
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

3. Représenter graphiquement la distribution dans un histogramme
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

4. Calculer et interpréter les quartiles
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


### Echantillon de taille 2000 suivant une loi normale de paramètre $N(\mu = 4, \sigma = 2)$. 

1. Simuler un échantillon de taille 2000 suivant une loi uniforme de paramètre $N(\mu = 4, \sigma = 2)$
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

2. Calculer la moyenne, le minimum, le maximum, la médiane et l'écart-type
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

3. Représenter graphiquement la distribution dans un histogramme
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

4. Calculer et interpréter les déciles {1,9} et les centiles {90,95,99}
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

### Echantillon de taille 2000 suivant une loi normale de paramètre $N(\mu = 4, \sigma = 6)$. 

1. Simuler un échantillon de taille 2000 suivant une loi uniforme de paramètre $N(\mu = 4, \sigma = 6)$
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

2. Calculer la moyenne, le minimum, le maximum, la médiane et l'écart-type
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

3. Représenter graphiquement la distribution dans un histogramme
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

4. Calculer et interpréter les déciles {1,9} et les centiles {90,95,99}
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

### Echantillon de taille 2000 suivant une loi normale de paramètre $N(\mu = 0, \sigma = 1)$. 

:bulb: [Petite illustration](https://upload.wikimedia.org/wikipedia/commons/thumb/7/74/Normal_Distribution_PDF.svg/1080px-Normal_Distribution_PDF.svg.png) des différentes distributions selon les paramètres de la loi normale.

1. Simuler un échantillon de taille 2000 suivant une loi uniforme de paramètre $N(\mu = 4, \sigma = 6)$
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

2. Calculer la moyenne, le minimum, le maximum, la médiane et l'écart-type
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

3. Représenter graphiquement la distribution dans un histogramme
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

4. Calculer et interpréter les déciles {1,9} et les centiles {90,95,99}
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

## Exercice 3 - Lecture de la table de la loi normale

### Mémo
| Nom de l'indicateur | Description    | Notation | Formule                          |
|---------------------|----------------|----------|----------------------------------|
| Probabilité cumulative inférieure | Probabilité que la variable aléatoire soit inférieure ou égale à une valeur donnée. | $P(X \leq x)$    | $P(X \leq x) = p$        |
| Probabilité cumulative supérieure | Probabilité que la variable aléatoire soit supérieure à une valeur donnée. | $P(X > x)$      | $P(X > x) = 1 - P(X \leq x) = p$   |

### Comprendre la table de la loi normale centrée réduite. 

<img src="./img/table_loi_normale.png" alt="" style="height: 600px;">

1. Quelle est la valeur théorique de $x$ telle que  $P(X \leq x) = 0.90$ ?
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

2. Quelle est la valeur théorique de $x$ telle que  $P(X \leq x) = 0.95$ ?
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

3. Quelle est la valeur théorique de $x$ telle que  $P(X \leq x) = 0.99$ ?
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

4. Quelle est la probabilité théorique $p$ telle que  $P(X \leq 1.96) = p$ ?
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

### Calculer la probabilité cumulative supérieure. 

1. Quelle est la valeur théorique de $x$ telle que  $P(X > x) = 0.90$ ?
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

2. Quelle est la valeur théorique de $x$ telle que  $P(X > x) = 0.05$ ?
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

3. Quelle est la probabilité théorique $p$ telle que  $P(X > 1.96) = p$ ?
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

Il existe de nombreuses lois de probabilités qui permettent de décrire des distributions avec d'autres formes ([plus d'infos](https://miro.medium.com/v2/resize:fit:962/1*lST5ngOvSMPqTTQCXxzneA.png)).
Ces lois sont très utilisés notamment dans les tests statistiques pour modéliser des phénomènes à partir de distributions issues de lois théoriques.
Dans ce chapitre, nous avons abordé uniquement des lois continues.
