# Adaptive AI-Driven Telescope Target Prioritization Framework
### Probabilistic Liquid Water Detection on Exoplanets Under Observation Constraints

> **Research Paper Title:** *Adaptive Reinforcement Learning-Based Telescope Target Prioritization for Probabilistic Liquid Water Detection on Exoplanets*

---

## Research Problem

Observatories such as JWST, Kepler, and TESS face a fundamental resource allocation problem: with **6,000+ confirmed exoplanets** and limited observation time, which targets should be prioritized to maximize the probability of detecting liquid water or habitable conditions?

This framework answers:
> *Which exoplanets should a telescope observe NEXT to maximize scientific return per observation hour?*

---

## Novel Contribution

| Existing Systems | This System |
|-----------------|-------------|
| Classify habitability | **Dynamically reprioritizes targets** |
| Predict water signatures | **Considers telescope constraints** |
| Rank planets statically | **Uses uncertainty-aware AI** |
| | **Optimizes scientific reward** |
| | **Adapts after each observation cycle** |

---

## System Architecture

```
Layer 1 — Data Acquisition
    NASA Exoplanet Archive TAP API (pscomppars, 6284 planets)
    Mass-radius imputation (Chen & Kipping 2017)
    Equilibrium temperature derivation

Layer 2 — Feature Engineering
    Habitable Zone boundaries (Kopparapu et al. 2014)  [label only]
    Earth Similarity Index (Schulze-Makuch et al. 2011) [label only]
    Observation feasibility: SNR proxy, distance penalty, detectability

Layer 3 — Weak Supervision Target
    priority_score = f_thermal^0.5 * f_rocky^0.3 * f_ESI^0.2

Layer 4 — ML Prioritization Models
    Random Forest  |  XGBoost  |  Gradient Boosting

Layer 5 — Uncertainty Estimation
    sigma_i = std(tree_1(x_i), ..., tree_N(x_i))
    scientific_gain = uncertainty * detectability

Layer 6 — Temporal Observation Simulation
    3 rounds x top-10 selection, uncertainty update, re-ranking
    vs. Random / Static ML baselines
```

---

## Key Design Decisions

| Decision | Approach | Reason |
|----------|----------|--------|
| Task type | **Regression (ranking)** not classification | Prioritization is ordinal, not binary |
| Data leakage | HZ/ESI **excluded** from ML features | Used only for weak supervision label |
| Target | Continuous `priority_score` in [0,1] | Ranking requires ordinal target |
| Observation | SNR proxy, distance penalty, detectability | Real telescope scheduling |
| Uncertainty | Ensemble tree variance | Research-grade, no BNN needed |
| Scientific gain | `uncertainty * detectability` | Information-theoretic scheduling |
| Metrics | NDCG, MAP, Spearman, Kendall tau | Ranking-appropriate evaluation |

---

## Scientific Formulations

**Habitable Zone (Kopparapu et al. 2014) — used for label construction only:**

$$S_{eff} = S_{eff\odot} + aT_* + bT_*^2 + cT_*^3 + dT_*^4, \quad d_{HZ} = \sqrt{L_\star / S_{eff}}$$

**Priority Score (weak supervision target):**

$$\text{priority\_score} = f_{\text{thermal}}^{0.5} \cdot f_{\text{rocky}}^{0.3} \cdot f_{\text{ESI}}^{0.2}$$

**Scientific Gain:**

$$\text{ScientificGain}_i = \sigma_i \times \text{Detectability}_i$$

**Stage 2 Dynamic Priority Formula (upcoming):**

$$\text{Priority} = \alpha H + \beta S + \gamma U + \delta O$$

**Stage 3 RL Reward (upcoming):**

$$R = \Delta\text{Confidence} + \lambda \times \text{WaterDetectionGain}$$

---

## ML Features (31 total — leakage-free)

| Category | Features |
|----------|----------|
| Planet physics | radius, mass, density, orbital period, semi-major axis, eccentricity, inclination, T_eq, insolation |
| Stellar physics | T_eff, radius, mass, luminosity, metallicity, log g, age |
| Distance/brightness | system distance, V-mag, J-mag |
| Observation feasibility | SNR proxy, distance penalty, transit depth, duration, detectability |
| Spectral type | star_O, star_B, star_A, star_F, star_G, star_K, star_M |

---

## Evaluation Metrics

```
Ranking Metrics (primary):
  NDCG@50      — quality of top-50 ranked list
  MAP@50       — precision of top-50 selection
  Spearman rho — rank correlation with ground truth
  Kendall tau  — concordance of pairwise rankings
  Regret@50    — scientific value missed vs ideal selection

Regression Metrics (secondary):
  R2, RMSE, MAE
```

---

## Project Structure

```
water/
├── habitability_predictor.ipynb   # Main notebook (run this on Colab)
├── src/
│   ├── __init__.py
│   ├── data_acquisition.py        # NASA API, feature engineering, target label
│   └── ml_pipeline.py             # Models, ranking metrics, uncertainty, simulation
├── data/
│   ├── exoplanets_processed.csv   # 5522 ML-ready planets
│   └── final_priority_ranking.csv # Final telescope target ranking (after training)
├── plots/                         # Generated EDA and performance plots
├── models/                        # Saved trained models
└── .gitignore
```

---

## Roadmap

| Stage | Status | Description |
|-------|--------|-------------|
| **Stage 1** | ✅ Complete | Static ML Habitability Predictor |
| **Stage 2** | Planned | Dynamic Prioritization Engine (adaptive weights) |
| **Stage 3** | Planned | RL Autonomous Scheduler (DQN/PPO) |

---

## Running on Google Colab

1. Open [colab.research.google.com](https://colab.research.google.com) → File → Open → GitHub
2. Enter `rushikesh-D69/water` and open `habitability_predictor.ipynb`
3. In Cell 0, set your `GIT_EMAIL` and `GIT_NAME`
4. Add your GitHub PAT to Colab Secrets (key: `GITHUB_TOKEN`)
5. Run all cells — results auto-push back to GitHub

## Running Locally

```bash
git clone https://github.com/rushikesh-D69/water.git
cd water
pip install xgboost lightgbm shap scipy scikit-learn matplotlib seaborn requests joblib
jupyter lab habitability_predictor.ipynb
```

---

## Data Source

- **NASA Exoplanet Archive** — [exoplanetarchive.ipac.caltech.edu](https://exoplanetarchive.ipac.caltech.edu)
- Table: `pscomppars` (Planetary Systems Composite Parameters)
- Access: TAP API with ADQL query
- Coverage: 6,284 confirmed exoplanets

---

## References

1. Kopparapu et al. (2013, 2014) — Habitable zone boundaries
2. Chen & Kipping (2017) — Mass-radius empirical relations
3. Schulze-Makuch et al. (2011) — Earth Similarity Index
4. Lundberg & Lee (2017) — SHAP feature attribution

---

## Target Publication Venues

- IEEE AI for Science workshops
- Springer Lecture Notes in Computer Science (AI/Astronomy)
- IJCAI / AAAI workshops on AI for Physical Sciences
- Monthly Notices of the Royal Astronomical Society (computational track)
