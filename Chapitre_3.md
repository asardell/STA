# Lien entre deux variables qualitatives

## Objectifs
Voici les objectifs de ce chapitre :
- [ ] Représenter graphiquement deux variables quantitatives
- [ ] Interpréter des profils lignes et colonnes
- [ ] Calculer et interpréter le V de Cramer

1. [Lien entre deux variables qualitatives](#lien-entre-deux-variables-qualitatives)
   1. [Objectifs](#objectifs)
   2. [Exercice 1 - Profils lignes vs Profils colonnes entre deux variables qualitatives](#exercice-1---profils-lignes-vs-profils-colonnes-entre-deux-variables-qualitatives)
      1. [Calculer les effectifs conditionnels et marginaux.](#calculer-les-effectifs-conditionnels-et-marginaux)
      2. [Calculer les fréquences conditionnelles et marginales.](#calculer-les-fréquences-conditionnelles-et-marginales)
      3. [Calculer les profils lignes.](#calculer-les-profils-lignes)
      4. [Calculer les profils colonnes.](#calculer-les-profils-colonnes)
   3. [Exercice 2 - Représentation graphique de deux variables qualitatives](#exercice-2---représentation-graphique-de-deux-variables-qualitatives)
      1. [Diagramme en barres non empilés.](#diagramme-en-barres-non-empilés)
      2. [Diagramme en barres empilés sur les effectifs.](#diagramme-en-barres-empilés-sur-les-effectifs)
      3. [Diagramme en barres empilés sur les profils lignes.](#diagramme-en-barres-empilés-sur-les-profils-lignes)
      4. [Diagramme en barres empilés sur les profils colonnes.](#diagramme-en-barres-empilés-sur-les-profils-colonnes)
   4. [Exercice 3 - Le V de Cramer](#exercice-3---le-v-de-cramer)
      1. [Mémo](#mémo)
      2. [Calculer les effectifs observés.](#calculer-les-effectifs-observés)
      3. [Calculer les effectifs théoriques.](#calculer-les-effectifs-théoriques)
      4. [Calculer les écarts entre effectifs observés et théoriques.](#calculer-les-écarts-entre-effectifs-observés-et-théoriques)
      5. [Calculer le V de Cramer.](#calculer-le-v-de-cramer)

Dans ce chapitre, nous allons utiliser le jeu de données Iris. Il est présent par défaut dans les environnements [R](https://rdrr.io/snippets/) et [Python](https://colab.research.google.com/). Il est aussi accessible dans le classeur Excel de ce repository.

:warning: le dataset peut avoir des différences selon le langage utilisé.

## Exercice 1 - Profils lignes vs Profils colonnes entre deux variables qualitatives

:warning: On choisit d'étudier le classe des passagers et leur survie (oui/non)

### Calculer les effectifs conditionnels et marginaux. 
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

### Calculer les fréquences conditionnelles et marginales. 
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

### Calculer les profils lignes. 
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

### Calculer les profils colonnes. 
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

## Exercice 2 - Représentation graphique de deux variables qualitatives

:warning: On choisit d'étudier le classe des passagers et leur survie (oui/non)

### Diagramme en barres non empilés. 

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

### Diagramme en barres empilés sur les effectifs. 

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

### Diagramme en barres empilés sur les profils lignes. 

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

### Diagramme en barres empilés sur les profils colonnes. 

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

## Exercice 3 - Le V de Cramer

:warning: On choisit d'étudier le classe des passagers et leur survie (oui/non)

### Mémo
| Nom de l'indicateur | Description    | Notation | Formule                          |
|---------------------|----------------|----------|----------------------------------|
| Effectifs observés     | Le nombre d'observations effectivement comptées pour chaque catégorie.      | $O_i$     | - |
| Effectifs théoriques   | Le nombre d'observations attendu pour chaque catégorie selon une hypothèse. | $E_i$      | $E_i = \frac{n \cdot f_i}{N}$      |
| Chi² locaux      | Mesure de l'écart entre les effectifs observés et théoriques pour chaque catégorie. | $\chi^2_i$     | $\chi^2_i = \frac{(O_i - E_i)^2}{E_i}$        |
| V de Cramer    | Mesure de l'association entre deux variables catégorielles.    | $V$    | $V = \sqrt{\frac{\chi^2}{n \cdot (k-1)}}$ où $\chi^2$ est la statistique chi-carré, $n$ est le nombre total d'observations, et $k$ est le nombre de catégories. |

### Calculer les effectifs observés. 

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

### Calculer les effectifs théoriques. 

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

### Calculer les écarts entre effectifs observés et théoriques. 

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

### Calculer le V de Cramer. 

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