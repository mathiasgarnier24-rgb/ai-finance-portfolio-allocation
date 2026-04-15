# Projet 5 - Modèle d'allocation de portefeuille par machine learning

---

# 1. Informations du Projet

- **Titre du projet :** Modèle d'allocation de portefeuille par machine learning
- **Nom du groupe :** Groupe 1
- **Membres du groupe :**
  - Étudiant 1 – Maël Pertuisot
  - Étudiant 2 – Valentin Martel
  - Étudiant 3 – Mathias Garnier

- **Cours :** AI In Finance
- **Encadrants :** Nicolas De Roux & Mohamed El Fakir
- **Date de soumission :** 20/04/2026

---

# 2. Description du Projet

L'allocation de portefeuille consiste à répartir un capital entre plusieurs actifs pour maximiser le rendement ajusté au risque. Les méthodes traditionnelles (Markowitz) reposent sur des estimations de rendements et de covariances souvent instables.

Ce projet explore l'utilisation du Machine Learning pour prédire les rendements et la volatilité de 50 actions du NASDAQ, puis utilise ces prédictions dans un cadre d'optimisation de portefeuille (Markowitz, maximum Sharpe). On compare les stratégies ML à un benchmark equal weight (1/N).

La **frontière efficiente de Markowitz** est tracée pour visualiser le positionnement de chaque stratégie dans l'espace rendement/risque.

Ce travail intéresse les gérants de fonds, analystes quantitatifs, et toute personne travaillant à l'intersection du data science et de la finance.

**Dataset de référence :** Stock Market Dataset (NASDAQ Universe) - données téléchargées via `yfinance`.

---

# 3. Objectif du Projet

Le projet vise à :

- Sélectionner 50 actions NASDAQ couvrant 8 secteurs
- Calculer des features : rendements, volatilité, corrélations, secteurs, indicateurs techniques
- Prédire les rendements hebdomadaires (next-week) avec des modèles ML
- Construire un portefeuille optimisé (Markowitz) à partir des prédictions
- Tracer la frontière efficiente et y positionner les différentes stratégies
- Évaluer la performance : ratio de Sharpe, max drawdown, turnover

---

# 4. Définition de la Tâche

- **Type de tâche :** Régression (prédiction de rendements continus)
- **Variables d'entrée :** Rendements passés (lags 1–5), moyennes mobiles, volatilité glissante, RSI, momentum, volume relatif, corrélation au marché, secteur (one-hot)
- **Variable cible :** Rendement cumulé sur les 5 prochains jours (next-week return)
- **Métriques d'évaluation :**
  - Prédiction : RMSE, MAE, R²
  - Portefeuille : Ratio de Sharpe, Max Drawdown, Turnover

---

# 5. Description du Dataset

## Vue d'ensemble

- **Nombre d'actions :** 50 tickers NASDAQ
- **Période :** 2018-01-01 à 2024-12-31 (~1750 jours de trading)
- **Nombre total d'observations :** ~80 000 (50 tickers × ~1600 jours après nettoyage)
- **Nombre de features :** ~20 features par action (+ one-hot secteur)
- **Source des données :** Yahoo Finance via `yfinance`

## Description des features

| Feature | Description | Type |
|---------|------------|------|
| ret_lag1 à ret_lag5 | Rendements passés (1 à 5 jours) | Numérique |
| ma_ratio | Ratio moyenne mobile 5j / 20j | Numérique |
| vol_10j, vol_20j | Volatilité glissante | Numérique |
| rsi | Relative Strength Index (14 jours) | Numérique |
| momentum_5j, momentum_20j | Rendement cumulé sur 5 et 20 jours | Numérique |
| vol_rel | Volume relatif (volume / moyenne 20j) | Numérique |
| corr_marche | Corrélation glissante avec le marché (30j) | Numérique |
| sect_* | Secteur de l'action (one-hot encoding) | Catégoriel |

## Variable cible

- Nom : `target`
- Signification : rendement cumulé sur les 5 prochains jours
- Valeurs continues, centrées autour de 0

## Distribution des données

- Les rendements hebdomadaires sont approximativement centrés autour de 0 avec des queues épaisses
- La volatilité varie fortement dans le temps (COVID-19 en 2020)
- Les corrélations inter-sectorielles varient dans le temps
- Pas de déséquilibre de classes (régression)

## Qualité des données

- Très peu de valeurs manquantes après nettoyage
- Les tickers avec plus de 5% de NaN sont exclus
- Pas de doublons

---

# 6. Prétraitement des données

- **Valeurs manquantes :** Tickers avec >5% NaN exclus, puis `dropna()` sur les features
- **Feature Engineering :** 20 features techniques créées à partir des prix et volumes bruts
- **Encodage catégoriel :** One-hot encoding des secteurs
- **Normalisation :** `StandardScaler` sur les features numériques (fit sur train uniquement)
- **Split temporel :** Train < 2023, Test ≥ 2023 (pas de mélange)

---

# 7. Approche de modélisation

## Modèles utilisés

1. **Régression Linéaire** - baseline
2. **Ridge Regression** (α=10) - régularisation L2
3. **Random Forest** (100 arbres, max_depth=6) - non-linéaire
4. **GRU** (hidden_dim=32, window=10) - deep learning séquentiel

## Choix du GRU

Le GRU (Gated Recurrent Unit) a été préféré au LSTM car :
- Il utilise 2 gates (update, reset) au lieu de 3, ce qui réduit le nombre de paramètres d'environ 25%
- Il est plus rapide à entraîner avec une précision comparable
- Il est particulièrement adapté aux horizons moyens (daily/weekly) selon la littérature
- Le gate `zt` contrôle le mélange entre ancien et nouveau état, et `rt` décide combien du passé utiliser pour le candidat

## Stratégie de modélisation

- Baseline linéaire puis complexité croissante
- Split temporel strict (pas de shuffle pour les séries temporelles)
- Random Forest avec profondeur limitée pour éviter l'overfitting
- GRU sur l'ensemble des 50 tickers pour capter les dépendances temporelles

## Stratégies de portefeuille

1. **Equal Weight (1/N)** - benchmark
2. **ML Long-Only** - allocation proportionnelle aux prédictions positives
3. **ML + Markowitz** - optimisation du ratio de Sharpe avec prédictions ML et covariance historique (60j), poids max 10% par action

## Frontière efficiente

La frontière efficiente de Markowitz est tracée sur la période de test avec :
- Les 50 actions individuelles colorées par secteur
- Le portefeuille GMV (Global Minimum Variance)
- Le portefeuille tangent (Maximum Sharpe)
- Le portefeuille Equal Weight
- Le portefeuille ML + Markowitz (poids moyens)
- La Capital Market Line (CML)

## Métriques d'évaluation

- **RMSE / MAE / R²** - qualité prédictive
- **Ratio de Sharpe** - rendement ajusté au risque
- **Max Drawdown** - risque extrême
- **Turnover** - coût de rééquilibrage

---

# 8. Structure du projet

```
project_portfolio/
├── project_portfolio.ipynb    # Notebook principal (code complet)
├── README.md                   # Ce fichier
```

Tout le code est dans un seul notebook.

---

# 9. Installation

Ouvrez le notebook directement dans votre navigateur :

1. Allez sur [colab.research.google.com](https://colab.research.google.com)
2. **Fichier → Ouvrir un notebook → onglet GitHub**
3. Collez l'URL du dépôt et sélectionnez le notebook `project_portfolio.ipynb`
4. **Exécution → Tout exécuter**
