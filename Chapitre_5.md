# Découvrir la loi uniforme et la loi normale

## Objectifs
Voici les objectifs de ce chapitre :
- [ ] Simuler une loi uniforme
- [ ] Simuler une loi normale
- [ ] Découvrir la loi normale centrée réduite
- [ ] Lire dans une table de loi normale


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
| Fonction de répartition    | Fonction qui donne la probabilité que la variable aléatoire soit inférieure ou égale à une valeur donnée. | $F(x)$           | $F(x) = P(X \leq x)$      |
| Fonction densité           | Fonction qui décrit la probabilité de la variable aléatoire continue.        | $f(x)$           | $f(x) = \frac{dF(x)}{dx}$ pour une variable continue          |
| Loi normale                | Distribution de probabilité continue symétrique, en forme de cloche, décrite par la moyenne et l'écart-type. | $N(\mu, \sigma)$ | $f(x) = \frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$   |

### Echantillon de taille 30 suivant une loi normale de paramètre $N(\mu = 4, \sigma^2 = 2)$. 

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


## Exercice 3 - Lecture de la table de la loi normale

### Mémo
| Nom de l'indicateur | Description    | Notation | Formule                          |
|---------------------|----------------|----------|----------------------------------|
| Loi normale                | Distribution de probabilité continue symétrique, en forme de cloche, décrite par la moyenne et l'écart-type. | $N(\mu, \sigma^2)$ | $f(x) = \frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$                         |
| Fonction de répartition    | Fonction qui donne la probabilité que la variable aléatoire soit inférieure ou égale à une valeur donnée. | $F(x)$           | $F(x) = P(X \leq x)$      |
| Fonction densité           | Fonction qui décrit la probabilité de la variable aléatoire continue.        | $f(x)$           | $f(x) = \frac{dF(x)}{dx}$ pour une variable continue          |

### Calculer des effectifs de chaque groupe. 

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
