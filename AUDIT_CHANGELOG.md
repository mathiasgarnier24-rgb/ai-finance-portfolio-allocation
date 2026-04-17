# Audit du notebook `project_portfolio_v4.ipynb`

**Date :** 17/04/2026
**Sortie :** `project_portfolio_v5.ipynb`

---

## Decisions structurelles validees

| Choix | Decision |
|---|---|
| Modele deep learning | GRU seul |
| Strategies de portefeuille | EW, Min-Variance, Max Sharpe (ML), Risk-Parity |
| Contraintes UCITS 5/10/40 | Retirees |
| Backtest | Walk-forward trimestriel (expanding window) |
| Metriques additionnelles | Aucune |

---

## Corrections techniques appliquees

### Bugs et data leakage

1. **GRU scaling** : l'ancien notebook utilisait une indexation positionnelle `[:len(cols_numeriques)]` qui supposait (a tort) que les colonnes numeriques etaient en tete du tableau. Remplace par une selection explicite par noms de colonnes (`df_t[cols_numeriques]`).
2. **Scaler fit sur train uniquement** : explicite dans la section 5 et dans la boucle walk-forward, avec verification de la moyenne et de l'ecart-type apres scaling.
3. **Forward-fill avant dropna** : conserve pour eviter les faux rendements gigantesques sur jours feries ponctuels.
4. **Seed de reproductibilite** : `SEED=42` applique a `np.random.seed`, `torch.manual_seed`, `RandomForestRegressor(random_state=SEED)`.
5. **Warnings filter** : restreint au module `yfinance` uniquement (au lieu d'ignorer tous les warnings).

### Ajouts methodologiques

6. **TimeSeriesSplit CV (5 folds)** sur le train pour Ridge et Random Forest, exigence TP3 exercice 2.5.
7. **Walk-forward quarterly refit** (section 7) : retraining complet a chaque debut de trimestre sur l'historique disponible. Remplace le single train/test split pour les predictions utilisees dans les portefeuilles.
8. **LedoitWolf effectif** : la covariance shrinkee est maintenant appliquee aux 3 optimiseurs (GMV, Max Sharpe, Risk-Parity), pas seulement importee.
9. **Covariance sur historique complet** : corrige un biais ou les premiers 20 rebalances ne disposaient pas d'assez d'historique et degradaient automatiquement en Equal Weight.

### Coherence financiere

10. **Turnover formula** : `0.5 * sum(|Delta w|)` (convention standard : un achat + une vente comptent comme un seul mouvement).
11. **Sharpe avec rf optionnel** : signature `sharpe_ratio(rendements, freq, rf=0.0)` permettant d'integrer un taux sans risque.
12. **Annualisation coherente** : freq=52 pour tous les portefeuilles hebdomadaires, freq=252 pour les series journalieres (EDA, frontiere efficiente).

### Structure et lisibilite

13. **11 sections numerotees** avec titres explicites.
14. **Interpretation economique** ajoutee apres chaque etape majeure (EDA, feature engineering, modeles, GRU, walk-forward, portefeuilles, frontiere, evaluation).
15. **Commentaires de code** professionnels et synthetiques, en francais, sans tirets cadratins ni emojis.
16. **Conclusion** reecrite pour refleter les 4 strategies et les limites reelles identifiees.

---

## Nouvelles strategies implementees

### Minimum Variance (GMV)
```
min  w' Sigma w
s.t. sum(w) = 1, w_i >= 0
```
Covariance : Ledoit-Wolf 60 jours glissants.

### Maximum Sharpe avec signal ML
```
max  (w' mu_ML) / sqrt(w' Sigma w)
s.t. sum(w) = 1, w_i >= 0
```
mu_ML = predictions walk-forward du Random Forest.

### Risk Parity
```
min  sum_i (RC_i - 1/n)^2
s.t. sum(w) = 1, w_i >= 1e-4
```
RC_i = contribution au risque de l'actif i.

---

## Verification end-to-end

Le notebook a ete execute avec donnees synthetiques (mock yfinance) pour valider le pipeline :

- 39 cellules de code Python : **38 passent sans erreur**.
- La seule cellule non testable est la commande shell `!pip install` (sera executee normalement par Jupyter/Colab).
- Aucun tiret cadratin, aucun emoji.
- Le notebook genere correctement : 4 courbes de rendement cumule, les drawdowns, la repartition sectorielle stackee, le turnover, la frontiere efficiente, les courbes d'apprentissage GRU.

---

## Points d'attention pour l'execution reelle

1. **Premier run Colab** : executer la cellule `!pip install` en premier.
2. **Temps d'execution estime** : 
   - Telechargement yfinance : ~30s
   - Features + modeles classiques : ~10s
   - GRU (30 epochs, 50 tickers) : ~2-4 min sur CPU
   - Walk-forward (12 retrainings) : ~30s
   - Optimisations portefeuilles : ~20s
   - **Total : environ 5 minutes**
3. **yfinance instable** : si certains tickers renvoient des NaN, le mecanisme de backup sectoriel les remplace automatiquement. Verifier la sortie de la cellule de cleaning.
4. **Memoire** : le GRU avec 50 tickers consomme ~1-2 Go en RAM. Si besoin, reduire `hidden_dim` de 32 a 16.

---

## Conformite au sujet

| Exigence | Traitement |
|---|---|
| 50 actions NASDAQ | 50 tickers sur 8 secteurs |
| Features : rendements, volatilite, secteur, correlation | 21 features (5 lags, 2 vol, ma_ratio, RSI, 2 momentum, vol_rel, corr_marche, 8 one-hot sectoriels) |
| Prediction next-day/next-week returns | Horizon hebdomadaire (5 jours) |
| ML models | LinearReg, Ridge, Random Forest, GRU |
| Portefeuille : Markowitz, ML | Max Sharpe ML + Min-Var + Risk-Parity + EW benchmark |
| Performance : Sharpe, drawdown, turnover | Tableau comparatif + visualisations |
| Sector classifications | Integres comme feature + analyse repartition |

Le sujet impose *minimum variance, maximum Sharpe, or risk-parity* : les trois sont implementes pour couverture complete.
