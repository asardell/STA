# Le test de corrélation

## Objectifs
Voici les objectifs de ce chapitre :
- [ ] Calculer et interpréter un test de corrélation de Pearson
- [ ] Différencier un test paramétrique et non paramétrique 
- [ ] Analyser la p-value
- [ ] Comprendre l'intérêt des hypothèses de départ et alternative

1. [Le test de corrélation](#le-test-de-corrélation)
   1. [Objectifs](#objectifs)
   2. [Exercice 1 - Coefficient de corrélation linéaire de Pearson](#exercice-1---coefficient-de-corrélation-linéaire-de-pearson)
      1. [Mémo](#mémo)
      2. [Reprendre la matrice de corrélation des variables quantitatives.](#reprendre-la-matrice-de-corrélation-des-variables-quantitatives)
   3. [Exercice 2 - Test statistiques avec $H\_0$ et $H\_1$](#exercice-2---test-statistiques-avec-h_0-et-h_1)
   4. [Exercice 3 - L'influence de la taille de l'échantillon](#exercice-3---linfluence-de-la-taille-de-léchantillon)
   5. [Exercice 4 - Cas de relation non linéaire mais monotone](#exercice-4---cas-de-relation-non-linéaire-mais-monotone)
   6. [Bonus - Cas de relation non linéaire et non monotone](#bonus---cas-de-relation-non-linéaire-et-non-monotone)

Dans ce chapitre, nous allons utiliser le jeu de données Iris. Il est présent par défaut dans les environnements [R](https://rdrr.io/snippets/) et [Python](https://colab.research.google.com/). Il est aussi accessible dans le classeur Excel de ce repository.

:warning: le dataset peut avoir des différences selon le langage utilisé.

## Exercice 1 - Coefficient de corrélation linéaire de Pearson

### Mémo
| Nom de l'indicateur | Description    | Notation | Formule                          |
|---------------------|----------------|----------|----------------------------------|
| Covariance   | Mesure de la tendance linéaire entre deux variables aléatoires. | $\text{cov}(X, Y)$ | $\text{cov}(X, Y) = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})$ |
| Coefficient de corrélation | Mesure de la force et de la direction de la relation linéaire entre deux variables aléatoires. | $r$ | $r = \frac{\text{cov}(X, Y)}{s_X \cdot s_Y}$ où $s_X$ et $s_Y$ sont les écart-types de $X$ et $Y$, respectivement. |
| Coefficient de détermination | Mesure la proportion de la variance de la variable dépendante expliquée par la variable indépendante dans un modèle de régression linéaire. | $R^2$ | $R^2 = r^2$ |
| Nuage de points           | Représentation graphique des paires de valeurs $(x_i, y_i)$ pour deux variables $X$ et $Y$. | - | - |
| Coefficient de corrélation de Pearson | Mesure de la force et de la direction de la relation linéaire entre deux variables continues. | $r_P$                | $r_P = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum (x_i - \bar{x})^2 \sum (y_i - \bar{y})^2}}$    |
| Coefficient de corrélation de Spearman | Mesure de la force et de la direction de la relation monotone entre deux variables ordinales ou continues. | $\rho$                | $\rho = 1 - \frac{6 \sum d_i^2}{n(n^2 - 1)}$ où $d_i$ est la différence entre les rangs de chaque paire de données. |
| Coefficient de corrélation de Kendall | Mesure de la force et de la direction de la relation monotone entre deux variables ordinales ou continues. | $\tau$                | $\tau = \frac{2 (C - D)}{n(n-1)}$ où $C$ est le nombre de paires concordantes et $D$ le nombre de paires discordantes. |
| Coefficient de Corrélation Maximal d'Information (MIC) | Mesure la force d'association entre deux variables, capable de capturer une large gamme de types de relations, y compris les relations linéaires, non linéaires, monotones et non monotones. | $MIC$ | Calculé via des méthodes d'information mutuelle. Par exemple, avec l'algorithme MINE. |

### Reprendre la matrice de corrélation des variables quantitatives. 
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


## Exercice 2 - Test statistiques avec $H_0$ et $H_1$

## Exercice 3 - L'influence de la taille de l'échantillon

## Exercice 4 - Cas de relation non linéaire mais monotone

## Bonus - Cas de relation non linéaire et non monotone
