# Lien entre deux variables quantitatives

## Objectifs
Voici les objectifs de ce chapitre :
- [ ] Représenter graphiquement deux variables quantitatives
- [ ] Calculer et interpréter le coefficient de corrélation linéaire
- [ ] Construire une matrice des corrélations
- [ ] Calculer et interpréter le coefficient de détermination

Dans ce chapitre, nous allons utiliser le jeu de données Iris. Il est présent par défaut dans les environnements [R](https://rdrr.io/snippets/) et [Python](https://colab.research.google.com/). Il est aussi accessible dans le classeur Excel de ce repository.

:warning: le dataset peuvent avoir des différences selon le langage utilisé.

## Exercice 1 - Coefficient de corrélation linéaire

### Mémo
| Nom de l'indicateur | Description    | Notation | Formule                          |
|---------------------|----------------|----------|----------------------------------|
| Covariance   | Mesure de la tendance linéaire entre deux variables aléatoires. | $\text{cov}(X, Y)$ | $\text{cov}(X, Y) = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})$ |
| Coefficient de corrélation | Mesure de la force et de la direction de la relation linéaire entre deux variables aléatoires. | $r$ | $r = \frac{\text{cov}(X, Y)}{s_X \cdot s_Y}$ où $s_X$ et $s_Y$ sont les écart-types de $X$ et $Y$, respectivement. |
| Coefficient de détermination | Mesure la proportion de la variance de la variable dépendante expliquée par la variable indépendante dans un modèle de régression linéaire. | $R^2$ | $R^2 = r^2$ |
| Nuage de points           | Représentation graphique des paires de valeurs $(x_i, y_i)$ pour deux variables $X$ et $Y$. | - | - |

#### Calculer la covariance entre deux variables quantitatives. 
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

#### Calculer la covariance manuellement. 
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

#### Calculer le coefficient de corrélation entre deux variables quantitatives. 
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

#### Calculer ce coefficient manuellement. 
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

#### Calculer du coefficient de détermination. 
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

#### Construire un nuage de points entre ces deux variables. 
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

#### Le Quartet d'Anscombe. 

:warning: Il est toujours important de visualiser ces données. Plus d'info avec le [Quartet d'Anscombe](https://blog.revolutionanalytics.com/2017/05/the-datasaurus-dozen.html)

## Exercice 2 - Matrice de corrélation

#### Calculer la matrice de corrélation des variables quantitatives. 
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