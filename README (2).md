# Projet 5 — Machine Learning pour l'Allocation de Portefeuille

---

# 1. Informations du Projet

- **Titre du projet :** Machine Learning pour l'Allocation de Portefeuille
- **Nom du groupe :** [Insérer le nom du groupe]
- **Membres du groupe :**
  - Étudiant 1 – [Nom]
  - Étudiant 2 – [Nom]
  - Étudiant 3 – [Nom]

- **Cours :** AI In Finance
- **Encadrants :** Nicolas De Roux & Mohamed El Fakir
- **Date de soumission :** Avril 2026

---

# 2. Description du Projet

L'allocation de portefeuille est un problème central en finance : comment répartir son capital entre plusieurs actifs pour maximiser le rendement tout en contrôlant le risque ? Traditionnellement, on utilise des méthodes comme l'optimisation de Markowitz, mais ces approches reposent sur des hypothèses fortes (rendements gaussiens, paramètres stables dans le temps).

Ce projet explore l'utilisation de méthodes de Machine Learning pour prédire les rendements futurs d'actions, puis utilise ces prédictions pour guider la construction d'un portefeuille. L'intérêt est de comparer une allocation guidée par le ML à un benchmark classique (equal weight).

Ce travail est pertinent pour les gérants de fonds, analystes quantitatifs, et toute personne intéressée par l'intersection entre data science et finance.

---

# 3. Objectif du Projet

Le projet vise à :

- Prédire les rendements journaliers d'un panier de 5 actions (AAPL, MSFT, JNJ, JPM, XOM) à partir d'indicateurs techniques
- Construire un portefeuille dont l'allocation est guidée par les prédictions des modèles ML
- Évaluer la performance de ce portefeuille face à un benchmark equal weight (1/N)

Un résultat réussi montrerait que le portefeuille ML obtient un meilleur ratio de Sharpe ou un drawdown plus faible que le benchmark.

---

# 4. Définition de la Tâche

- **Type de tâche :** Régression (prédiction de rendements continus)
- **Variables d'entrée :** Rendements passés (lags 1–5), moyennes mobiles (5j et 20j), ratio MA, volatilité glissante (10j), RSI (14j), momentum (5j)
- **Variable cible :** Rendement du jour suivant (return t+1)
- **Métriques d'évaluation :**
  - Pour la prédiction : RMSE, MAE, R²
  - Pour le portefeuille : Ratio de Sharpe, Max Drawdown, Rendement annualisé

---

# 5. Description du Dataset

## Vue d'ensemble

- **Nombre d'observations :** ~2500 jours de trading par action (2015–2024)
- **Nombre de features :** 10 features techniques par action
- **Variable cible :** Rendement journalier du lendemain
- **Source des données :** Yahoo Finance via la librairie `yfinance`

## Description des features

| Feature | Description | Type |
|---------|------------|------|
| return_lag1 à return_lag5 | Rendements passés (1 à 5 jours) | Numérique |
| ma5 | Moyenne mobile 5 jours | Numérique |
| ma20 | Moyenne mobile 20 jours | Numérique |
| ma_ratio | Ratio MA5 / MA20 | Numérique |
| volatilite_10j | Écart-type glissant sur 10 jours | Numérique |
| rsi | Relative Strength Index (14 jours) | Numérique |
| momentum_5j | Rendement cumulé sur 5 jours | Numérique |

## Variable cible

- Nom : `target`
- Signification : rendement journalier du jour suivant (en pourcentage)
- Valeurs continues, centrées autour de 0

## Types de données

Toutes les variables sont numériques (séries temporelles financières).

## Distribution des données

- Les rendements sont approximativement centrés autour de 0 avec des queues épaisses (fat tails)
- La volatilité varie fortement dans le temps (hétéroscédasticité), notamment pendant le COVID-19 (2020)
- Les corrélations entre actions varient selon les secteurs

## Qualité des données

- Très peu de valeurs manquantes (données de marché fiables)
- Quelques jours sans trading (weekends, jours fériés) déjà exclus par Yahoo Finance
- Pas de doublons

---

# 6. Prétraitement des données

- **Valeurs manquantes :** Supprimées avec `dropna()` après la création des features (les premières lignes n'ont pas assez d'historique pour les moyennes mobiles et le RSI)
- **Feature Engineering :** Création de 10 features techniques à partir des prix bruts (détail dans la section 4)
- **Normalisation :** `StandardScaler` appliqué sur les features avant entraînement des modèles (fit uniquement sur le train)
- **Split temporel :** 80% train / 20% test, sans mélange (respecter l'ordre chronologique)

---

# 7. Approche de modélisation

## Modèles utilisés

1. **Régression Linéaire** — baseline simple
2. **Ridge Regression** — régression avec régularisation L2 pour éviter le surapprentissage
3. **Random Forest Regressor** — modèle non-linéaire basé sur des arbres de décision
4. **LSTM** — réseau de neurones récurrent pour capter les dépendances temporelles

## Stratégie de modélisation

- On commence par un modèle baseline (régression linéaire) pour avoir un point de comparaison
- On ajoute de la régularisation (Ridge) pour stabiliser les coefficients
- On teste un modèle non-linéaire (Random Forest) avec profondeur limitée (max_depth=5) pour éviter l'overfitting
- On entraîne un LSTM sur AAPL pour voir si les dépendances temporelles apportent un gain
- Le split est toujours temporel (pas de cross-validation aléatoire sur séries temporelles)

## Métriques d'évaluation

- **RMSE** : erreur moyenne en unités originales, sensible aux grosses erreurs
- **MAE** : erreur absolue moyenne, plus robuste aux outliers
- **R²** : proportion de variance expliquée (souvent faible en finance)
- **Ratio de Sharpe** : rendement ajusté au risque du portefeuille
- **Max Drawdown** : perte maximale depuis un pic, mesure du risque extrême
- **Rendement annualisé** : performance globale du portefeuille

Le RMSE et le R² évaluent la qualité prédictive des modèles, tandis que le Sharpe et le Drawdown évaluent l'utilité économique des prédictions.

---

# 8. Structure du projet

```
projet5_portfolio/
├── project5_portfolio.ipynb    # Notebook principal (code complet)
├── README.md                   # Ce fichier
```

Tout le code est dans un seul notebook pour simplifier.

---

# 9. Installation

```bash
pip install numpy pandas matplotlib seaborn yfinance scikit-learn torch
```

Puis ouvrir le notebook `project5_portfolio.ipynb` dans Jupyter ou Google Colab.
