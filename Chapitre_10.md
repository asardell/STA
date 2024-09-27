# Démarche d'une modélisation - Cas de la régression linéaire

## Objectifs
Voici les objectifs de ce chapitre :
- [ ] Comprendre l'importance de l'échantillon d'apprentissage et test
- [ ] Calculer et interpréter une régression linaire simple et multiple
- [ ] Calculer les performances d'un modèle

1. [Démarche d'une modélisation - Cas de la régression linéaire](#démarche-dune-modélisation---cas-de-la-régression-linéaire)
   1. [Objectifs](#objectifs)
   2. [Exercice 1 : Régression linéaire simple](#exercice-1--régression-linéaire-simple)
      1. [Mémo](#mémo)
      2. [Sepal Length en fonction de Sepal Width](#sepal-length-en-fonction-de-sepal-width)
      3. [Sepal Length en fonction de Petal Width](#sepal-length-en-fonction-de-petal-width)
   3. [Bonus : Régression linéaire multiple](#bonus--régression-linéaire-multiple)

## Exercice 1 : Régression linéaire simple

### Mémo
| Nom de l'indicateur | Description    | Notation | Formule                          |
|---------------------|----------------|----------|----------------------------------|
| **Échantillon d'apprentissage (train set)**   | Sous-ensemble de données utilisé pour entraîner le modèle.                                                      | -        | -       |
| **Échantillon de test (test set)**            | Sous-ensemble de données utilisé pour évaluer la performance du modèle entraîné.                                | -        | -       |
| **Erreur quadratique moyenne (MSE)**          | Mesure de la qualité d'un estimateur en calculant la moyenne des carrés des erreurs ou résidus.                  | MSE      | $\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$ |
| **Erreur quadratique moyenne racine (RMSE)**  | Racine carrée de la MSE, mesurant la différence moyenne entre les valeurs observées et prédites.                | RMSE     | $\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$ |
| **Coefficient de corrélation** | Mesure de la force et de la direction de la relation linéaire entre deux variables aléatoires. | $r$ | $r = \frac{\text{cov}(X, Y)}{s_X \cdot s_Y}$ où $s_X$ et $s_Y$ sont les écart-types de $X$ et $Y$, respectivement. |
| **Coefficient de détermination (R²)**         | Proportion de la variance totale des variables dépendantes expliquée par le modèle de régression.                | $R^2$ | $R^2 = r^2$ |
| **Pente (Coefficient de régression)**         | Mesure de la force et de la direction de la relation linéaire entre deux variables dans un modèle de régression. | $\beta$  | $\beta = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2}$ |
| **Intercept**                                 | Valeur de la variable dépendante lorsque toutes les variables indépendantes sont nulles (point où la droite coupe l'axe Y). | $\alpha$ | $\alpha = \bar{y} - \beta \bar{x}$ |

Dans cet exemple, on modélise Sepal Length en fonction de Sepal Width

### Sepal Length en fonction de Sepal Width

1. Construire l'échantillon train/test avec 30% de test.

<details>
<summary>R</summary>

```r
# Charger les packages nécessaires
library(datasets)
library(caret)
library(ggplot2)

# Charger le jeu de données Iris
data(iris)

# Sélectionner les variables
X <- iris$Sepal.Width
y <- iris$Sepal.Length

# Diviser les données en échantillon d'apprentissage et de test (70% / 30%)
set.seed(42)
trainIndex <- createDataPartition(y, p = .7, list = FALSE, times = 1)
X_train <- X[trainIndex]
y_train <- y[trainIndex]
X_test <- X[-trainIndex]
y_test <- y[-trainIndex]
```
</details>

<details>
<summary>Python</summary>

```python
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Charger le dataset Iris
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

# Sélectionner les variables
X = iris_df[['sepal_width']]
y = iris_df['sepal_length']

# Diviser les données en échantillon d'apprentissage et de test (70% / 30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```
</details>

2. Afficher le nuage de point de la relation en les deux variables

<details>
<summary>R</summary>

```r
# Nuage de points pour les données d'entraînement
plot(X_train, y_train, 
     main = "Nuage de points et droite de régression (Iris)", 
     xlab = "Longueur des sépales (cm)", 
     ylab = "Largeur des sépales (cm)", 
     pch = 19, col = "black")

# Ajouter une légende
legend("topright", legend = "Valeurs observées", col = "black", pch = 19)
```
</details>

<details>
<summary>Python</summary>

```python
# Taille de la figure
plt.figure(figsize=(10, 6))

# Nuage de points pour les données d'entraînement
plt.scatter(X_train, y_train, color='black', label='Valeurs observées')

# Ajouter des labels et une légende
plt.xlabel('Longueur des sépales (cm)')
plt.ylabel('Largeur des sépales (cm)')
plt.title('Nuage de points et droite de régression (Iris)')
plt.legend()

# Afficher le graphique
plt.show()
```
</details>

3. Calculer le coefficient de corrélation

<details>
<summary>R</summary>

```r
correlation <- cor(X_train, y_train)
print(correlation)
```
</details>

<details>
<summary>Python</summary>

```python
import numpy as np
# Calculer le coefficient de corrélation avec NumPy
correlation = np.corrcoef(X.values.flatten(), y)[0, 1]
print(f"Coefficient de corrélation (NumPy) entre la longueur et la largeur des sépales : {correlation:.2f}")

```
</details>

4. Construire le modèle de régression linéaire.

<details>
<summary>R</summary>

```r
# Construire le modèle de régression linéaire
model <- lm(Sepal.Length ~ Sepal.Width, data = iris[trainIndex,])
```
</details>

<details>
<summary>Python</summary>

```python
# Construire le modèle de régression linéaire
model = LinearRegression()
model.fit(X_train, y_train)
```
</details>

5. Analyser les coéfficients du modèle.

<details>
<summary>R</summary>

```r
# Analyser les coefficients du modèle
summary(model)
```
</details>

<details>
<summary>Python</summary>

```python
# Analyser les coefficients du modèle
intercept = model.intercept_
coefficient = model.coef_[0]
print(f'Intercept: {intercept}')
print(f'Coefficient: {coefficient}')
```
</details>

6. Prédire sur les données test.

<details>
<summary>R</summary>

```r
# Prédire sur les données test
y_pred <- predict(model, newdata = X_test)
```
</details>

<details>
<summary>Python</summary>

```python
# Prédire sur les données test
y_pred = model.predict(X_test)
```
</details>

8. Représenter graphiquement la variables explicative et la variables à expliquer avec la droite de régression

<details>
<summary>R</summary>

```r
# Nuage de points pour les données d'entraînement
plot(X_train, y_train, 
     main = "Nuage de points et droite de régression (Iris)", 
     xlab = "Longueur des sépales (cm)", 
     ylab = "Largeur des sépales (cm)", 
     pch = 19, col = "black")

# Tracer la droite de régression
# Supposons que y_pred est préalablement calculé à partir d'un modèle
abline(lm(y_train ~ X_train), col = "blue")

# Ajouter une légende
legend("topright", legend = c("Valeurs observées", "Droite de régression"), 
       col = c("black", "blue"), pch = c(19, NA), lty = c(NA, 1))
```
</details>

<details>
<summary>Python</summary>

```python
# Taille de la figure
plt.figure(figsize=(10, 6))

# Nuage de points pour les données d'entraînement
plt.scatter(X_train, y_train, color='black', label='Valeurs observées')

# Tracer la droite de régression
plt.plot(X_test, y_pred, color='blue', label='Droite de régression')

# Ajouter des labels et une légende
plt.xlabel('Longueur des sépales (cm)')
plt.ylabel('Largeur des sépales (cm)')
plt.title('Nuage de points et droite de régression (Iris)')
plt.legend()

# Afficher le graphique
plt.show()
```
</details>


9.  Représenter graphiquement les valeurs prédites et observés.

<details>
<summary>R</summary>

```r
# Nuage de points pour les valeurs observées et prédites
plot(y_test, y_pred, 
     main = "Nuage de Points: Valeurs Observées vs Prédictions", 
     xlab = "Valeurs Observées", 
     ylab = "Valeurs Prédites", 
     pch = 19, col = "black")

# Tracer la droite d'équation y = x
max_val <- max(max(y_test), max(y_pred))
abline(a = 0, b = 1, col = "blue", lty = 2)

# Définir les limites des axes
xlim <- c(0, max_val)
ylim <- c(0, max_val)
xlim(xlim)
ylim(ylim)

# Ajouter une légende
legend("topleft", legend = "y = x (Droite de référence)", col = "blue", lty = 2)
```
</details>

<details>
<summary>Python</summary>

```python
# Taille de la figure
plt.figure(figsize=(10, 6))

# Nuage de points pour les valeurs observées et prédites
plt.scatter(y_test, y_pred, color='black', label='Valeurs Observées vs Prédictions')

# Tracer la droite d'équation y = x
max_val = max(y_test.max(), y_pred.max())
plt.plot([0, max_val], [0, max_val], color='blue', linestyle='--', label='y = x (Droite de référence)')

# Définir les limites des axes
plt.xlim(0, max_val)
plt.ylim(0, max_val)

# Assurer que les axes ont la même échelle
plt.gca().set_aspect('equal', adjustable='box')

# Ajouter des labels et une légende
plt.xlabel('Valeurs Observées')
plt.ylabel('Valeurs Prédites')
plt.title('Nuage de Points: Valeurs Observées vs Prédictions')
plt.legend()

# Afficher le graphique
plt.show()
```
</details>

10. Calculer les métriques pour évaluer le modèle (coefficient de corrélation, MSE, RMSE).

<details>
<summary>R</summary>

```r
# Calculer les métriques pour évaluer le modèle
mse <- mean((y_test - y_pred)^2)
rmse <- sqrt(mse)
correlation <- cor(y_test, y_pred)
r_squared <- summary(model)$r.squared

cat(sprintf('MSE: %f\n', mse))
cat(sprintf('RMSE: %f\n', rmse))
cat(sprintf('Coefficient de détermination (R²): %f\n', r_squared))
cat(sprintf('Coefficient de corrélation: %f\n', correlation))
```
</details>

<details>
<summary>Python</summary>

```python
# Calculer les métriques pour évaluer le modèle
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
correlation = np.corrcoef(y_test, y_pred)[0, 1]

print(f'MSE: {mse}')
print(f'RMSE: {rmse}')
print(f'Coefficient de détermination (R²): {r2}')
print(f'Coefficient de corrélation: {correlation}')
```
</details>


### Sepal Length en fonction de Petal Width



## Bonus : Régression linéaire multiple

Cas pratique pour prédire Sepal Length en fonction des 3 autres variables quantitatives.

<details>
<summary>R</summary>

```r
# Charger les packages nécessaires
library(datasets)
library(caret)
library(ggplot2)

# Charger le dataset Iris
data(iris)

# Sélectionner les variables
X <- iris[, c('Sepal.Width', 'Petal.Length', 'Petal.Width')]
y <- iris$Sepal.Length

# Diviser les données en échantillon d'apprentissage et de test (70% / 30%)
set.seed(42)
trainIndex <- createDataPartition(y, p = .7, list = FALSE, times = 1)
X_train <- X[trainIndex, ]
y_train <- y[trainIndex]
X_test <- X[-trainIndex, ]
y_test <- y[-trainIndex]

# Construire le modèle de régression linéaire
model <- lm(Sepal.Length ~ Sepal.Width + Petal.Length + Petal.Width, data = iris[trainIndex, ])

# Analyser les coefficients du modèle
summary(model)

# Prédire sur les données test
y_pred <- predict(model, newdata = X_test)

# Représenter graphiquement les valeurs prédites et observées
df_test <- data.frame(Observed = y_test, Predicted = y_pred)
ggplot(df_test, aes(x = Predicted, y = Observed)) +
  geom_point(color = 'black') +
  geom_abline(slope = 1, intercept = 0, color = 'blue') +
  labs(x = 'Valeurs prédites', y = 'Valeurs observées', 
       title = 'Régression Linéaire - Sepal Length vs Sepal Width, Petal Length, Petal Width') +
  theme_minimal()

# Calculer les métriques pour évaluer le modèle
mse <- mean((y_test - y_pred)^2)
rmse <- sqrt(mse)
r2 <- summary(model)$r.squared
correlation <- cor(y_test, y_pred)

cat(sprintf('MSE: %f\n', mse))
cat(sprintf('RMSE: %f\n', rmse))
cat(sprintf('Coefficient de détermination (R²): %f\n', r2))
cat(sprintf('Coefficient de corrélation: %f\n', correlation))
```
</details>


<details>
<summary>Python</summary>

```python
# Sélectionner les variables
X = iris_df[['sepal_width', 'petal_length', 'petal_width']]
y = iris_df['sepal_length']

# Diviser les données en échantillon d'apprentissage et de test (70% / 30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Construire le modèle de régression linéaire
model = LinearRegression()
model.fit(X_train, y_train)

# Analyser les coefficients du modèle
intercept = model.intercept_
coefficient = model.coef_
print(f'Intercept: {intercept}')
print(f'Coefficient: {coefficient}')

# Prédire sur les données test
y_pred = model.predict(X_test)

# Représenter graphiquement les valeurs prédites et observées
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, y_test, color='black', label='Valeurs observées')
plt.plot(y_pred, y_pred, color='blue', linewidth=2, label='Valeurs prédites')
plt.show()

# Calculer les métriques pour évaluer le modèle
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
correlation = np.corrcoef(y_test, y_pred)[0, 1]

print(f'MSE: {mse}')
print(f'RMSE: {rmse}')
print(f'Coefficient de détermination (R²): {r2}')
print(f'Coefficient de corrélation: {correlation}')
```
</details>