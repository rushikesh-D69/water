# Adaptive AI-Driven Telescope Target Prioritization Framework
### Probabilistic Liquid Water Detection on Exoplanets Under Observation Constraints

> **Research Paper Title:** *Adaptive Uncertainty-Aware Exoplanet Observation Scheduling Under Telescope Resource Constraints*

[![GitHub](https://img.shields.io/badge/GitHub-rushikesh--D69%2Fwater-22b5a0?style=flat-square&logo=github)](https://github.com/rushikesh-D69/water)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)](https://python.org)
[![Stage](https://img.shields.io/badge/Stage%202-In%20Progress-f0a500?style=flat-square)](#roadmap)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?style=flat-square&logo=streamlit)](#streamlit-dashboard)

---

## Research Question

> *Can an AI-driven scheduler optimize exoplanet observation allocation more effectively than static prioritization approaches?*

Stage 1 answered: **"Which exoplanets are promising?"**
Stage 2 answers: **"Given limited telescope resources, which planets should be observed NEXT to maximize scientific gain?"**

---

## System Architecture

```
Stage 1 — Probabilistic Prioritization (Complete)
──────────────────────────────────────────────────
  NASA TAP API (6,284 planets)
      ↓
  Feature Engineering (31 features, leakage-free)
      ↓
  Weak Supervision Target: priority = (H+ε)^0.5 (R+ε)^0.3 (E+ε)^0.2
      ↓
  Ensemble ML: RF | XGBoost | GBM | LightGBM
      ↓
  Uncertainty: sigma_i = std(tree predictions)
      ↓
  Stage 1 Output: ranked priority scores + uncertainty map

Stage 2 — Adaptive Observation Scheduling Engine (In Progress)
───────────────────────────────────────────────────────────────
  Constraint Engine:  AR(1) weather, time budget, visibility, cost
      ↓
  Scientific Gain:    Gain = α_t·U + β_t·P + γ·D  (β decays over time)
      ↓
  5 Schedulers:       Static | Detectability | Uncertainty | Adaptive | Oracle
      ↓
  Observation Simulator: SNR + weather + detectability-dependent noise
      ↓
  Adaptive Loop:      30 rounds × 10 planets, uncertainty update, re-ranking
      ↓
  Evaluation:         7 metrics + Oracle regret + Campaign Diversity
      ↓
  Streamlit Dashboard: 5 live panels
```

---

## Novel Contributions

| Existing Systems | This Framework |
|-----------------|----------------|
| Habitability classification | **Dynamic reprioritization** each round |
| Static ML ranking | **Adaptive scheduler** with time-decaying exploration |
| No resource model | **AR(1) weather + time budget + switching cost** |
| No upper bound | **OracleScheduler** for absolute regret measurement |
| Simple ranking metrics | **Campaign Diversity Score** (5-dimensional) |
| No feedback loop | **Closed-loop adaptive astronomy** |

---

## Stage 1 Results

### ML Performance (5,522 Exoplanets)

| Model | NDCG@50 | Spearman ρ | Regret@50 | R² |
|-------|---------|------------|-----------|-----|
| LightGBM | **0.9890** | **0.9850** | **0.0096** | **0.9709** |
| XGBoost | 0.9905 | 0.9850 | 0.0106 | 0.9709 |
| Random Forest | 0.9770 | 0.9719 | 0.0244 | 0.9428 |
| Gradient Boosting | 0.9754 | 0.9836 | 0.0296 | 0.9640 |

### Stage 1 Plots

<table>
<tr>
<td><img src="plots/priority_score_distribution.png" width="400"/><br><sub>Priority Score Distribution (quantile-normalized, uniform [0,1])</sub></td>
<td><img src="plots/feature_importance.png" width="400"/><br><sub>Feature Importance — top 20 astrophysical drivers</sub></td>
</tr>
<tr>
<td><img src="plots/shap_lightgbm.png" width="400"/><br><sub>SHAP Analysis — LightGBM (best model)</sub></td>
<td><img src="plots/predicted_vs_actual.png" width="400"/><br><sub>Predicted vs Actual Priority Score</sub></td>
</tr>
<tr>
<td><img src="plots/ranking_metrics.png" width="400"/><br><sub>Ranking Metrics Comparison (NDCG, Spearman, Kendall)</sub></td>
<td><img src="plots/uncertainty_analysis.png" width="400"/><br><sub>Prediction Uncertainty Distribution (tree variance)</sub></td>
</tr>
<tr>
<td><img src="plots/temporal_simulation.png" width="400"/><br><sub>Stage 1 Temporal Simulation — 3 rounds, uncertainty update</sub></td>
<td><img src="plots/observability_analysis.png" width="400"/><br><sub>Observability Analysis — detectability vs priority</sub></td>
</tr>
<tr>
<td><img src="plots/cv_spearman.png" width="400"/><br><sub>Cross-Validation Spearman ρ (5-fold)</sub></td>
<td><img src="plots/feature_correlations.png" width="400"/><br><sub>Feature Correlation Matrix (31 features)</sub></td>
</tr>
</table>

---

## Stage 2 — Adaptive Scheduling Engine

### 5 Schedulers

| Scheduler | Strategy | Role |
|-----------|----------|------|
| **Static Priority** | Always top priority_score | Baseline 1 |
| **Detectability Greedy** | Always top detectability | Baseline 2 |
| **Uncertainty Greedy** | Always highest uncertainty | Baseline 3 |
| **Adaptive Scheduler** | `Utility = (Gain × F) / Cost` | **Our method** |
| **Oracle** | Perfect future knowledge | Upper bound for regret |

### Key Formulas

**Scientific Gain with time-decaying exploration:**
```
Gain_i = α_t · U_i + β_t · P_i + γ · D_i

β_t = β_0 · exp(-t / τ)    ← early exploration → late exploitation
```

**Scheduling Utility:**
```
Utility_i = (Gain_i × Feasibility_i) / Cost_i
```

**Oracle Regret:**
```
Regret@t = (Oracle_cumgain_t − Scheduler_cumgain_t) / Oracle_cumgain_t
```

**Observation Noise (Issue 4 — realistic):**
```
σ_new = σ_old × (0.4 + 0.4·D + 0.2·w) / √n_obs
```

**AR(1) Weather Model:**
```
w_t = ρ · w_{t-1} + (1-ρ) · Beta(5,2) + noise,  ρ = 0.65
```

### Campaign Diversity Metric (5 dimensions)

| Dimension | Measure |
|-----------|---------|
| Stellar Type Entropy | Shannon H over OBAFGKM distribution |
| Temperature Diversity | std(T_eq) / max(T_eq) |
| Orbital Diversity | std(log P) / mean(log P) |
| Mass Diversity | std(log M) / mean(log M) |
| Distance Coverage | std(d) / median(d) |

---

## Streamlit Dashboard

**5 interactive panels:**

| Panel | Content |
|-------|---------|
| Telescope Queue | Top-20 targets, priority × detectability scatter |
| Priority Evolution | Mean priority, uncertainty, cumulative gain, weight decay |
| Uncertainty Heatmap | T_eq vs Radius space — observed (stars) vs unobserved |
| Campaign Timeline | Gantt-style Plotly schedule with weather opacity encoding |
| Metrics Dashboard | Comparison table, live regret vs Oracle, all generated plots |

**Run locally:**
```bash
pip install streamlit plotly
streamlit run dashboard/app.py
```

**Run in Colab:**
```python
!pip install -q streamlit plotly
!streamlit run dashboard/app.py &
from google.colab.output import eval_js
print(eval_js('google.colab.kernel.proxyPort(8501)'))
```

---

## Scientific Formulations

**Priority Score (weak supervision, v2.1):**

$$\text{priority} = \frac{(H+\varepsilon)^{0.5}(R+\varepsilon)^{0.3}(E+\varepsilon)^{0.2} + \text{diversity nudges}}{1} \xrightarrow{\text{quantile}} \mathcal{U}[0,1]$$

**Habitable Zone (Kopparapu et al. 2014):**

$$S_{eff} = S_{eff\odot} + aT_* + bT_*^2 + cT_*^3 + dT_*^4, \quad d_{HZ} = \sqrt{L_\star / S_{eff}}$$

**Uncertainty (tree variance):**

$$\sigma_i = \text{std}\left(\hat{y}^{(1)}_i, \ldots, \hat{y}^{(N)}_i\right)$$

**Stage 3 RL Reward (upcoming):**

$$R_t = \Delta\text{InformationGain}_t - \lambda \cdot \text{ObservationCost}_t$$

---

## ML Features (31 total — leakage-free)

| Category | Features |
|----------|----------|
| Planet physics | radius, mass, density, orbital period, semi-major axis, eccentricity, inclination, T_eq, insolation |
| Stellar physics | T_eff, radius, mass, luminosity, metallicity, log g, age |
| Distance/brightness | system distance, V-mag, J-mag |
| Observation feasibility | SNR proxy, distance penalty, transit depth, duration, detectability |
| Spectral type (one-hot) | O, B, A, F, G, K, M |

> **Leakage prevention:** `hz_factor`, `esi`, `_diag` components excluded from ML features — used only in weak supervision label construction.

---

## Project Structure

```
water/
├── habitability_predictor.ipynb   # Stage 1: Main Colab notebook
├── stage2_pipeline.ipynb          # Stage 2: Scheduling campaign notebook
├── src/
│   ├── data_acquisition.py        # NASA API, feature engineering, v2.1 priority score
│   ├── ml_pipeline.py             # Ensemble models, ranking metrics, uncertainty
│   ├── constraint_engine.py       # AR(1) weather, time budget, visibility, cost
│   ├── observation_simulator.py   # SNR/weather/detectability-dependent noise model
│   ├── scheduler.py               # 5 schedulers + OracleScheduler + campaign runner
│   └── evaluation.py              # 7 metrics, comparison table, 7 plots
├── dashboard/
│   ├── app.py                     # Streamlit 5-panel dashboard
│   └── requirements.txt
├── data/
│   ├── exoplanets_processed.csv   # 5,522 ML-ready planets
│   ├── final_priority_ranking.csv # Ranked telescope targets
│   └── stage2_*.csv               # Stage 2 campaign results
├── plots/                         # All generated plots (Stage 1 + Stage 2)
├── models/                        # Saved trained models
├── report/
│   ├── main.tex                   # Full LaTeX technical report
│   └── references.bib
└── .gitignore
```

---

## Roadmap

| Stage | Status | Description |
|-------|--------|-------------|
| **Stage 1** | ✅ Complete | Static ML Ranking — 4 models, SHAP, uncertainty, temporal simulation |
| **Stage 2** | 🔄 In Progress | Adaptive Scheduling — 5 schedulers, Oracle regret, Campaign Diversity, Streamlit |
| **Stage 3** | Planned | RL Scheduler — PPO/DQN (Stable-Baselines3), MDP formulation |

---

## Running on Google Colab

**Stage 1:**
1. Open `habitability_predictor.ipynb` via GitHub in Colab
2. Add `GITHUB_TOKEN` to Colab Secrets
3. Run all cells — results auto-push to GitHub

**Stage 2:**
1. Open `stage2_pipeline.ipynb` via GitHub in Colab
2. Run all cells — campaigns, evaluation, plots auto-generated
3. Launch dashboard: `!streamlit run dashboard/app.py &`

**Local setup:**
```bash
git clone https://github.com/rushikesh-D69/water.git
cd water
pip install xgboost lightgbm shap scipy scikit-learn matplotlib seaborn \
            requests joblib streamlit plotly
jupyter lab stage2_pipeline.ipynb
```

---

## Data Source

- **NASA Exoplanet Archive** — [exoplanetarchive.ipac.caltech.edu](https://exoplanetarchive.ipac.caltech.edu)
- Table: `pscomppars` (Planetary Systems Composite Parameters)
- Access: TAP API with ADQL
- Coverage: 6,284 confirmed exoplanets → 5,522 after processing

---

## References

1. Kopparapu et al. (2013, 2014) — Habitable zone boundaries
2. Chen & Kipping (2017) — Mass-radius empirical relations
3. Schulze-Makuch et al. (2011) — Earth Similarity Index (ESI)
4. Lundberg & Lee (2017) — SHAP unified feature attribution
5. Chen & Guestrin (2016) — XGBoost
6. Ke et al. (2017) — LightGBM
7. Breiman (2001) — Random Forests
8. Järvelin & Kekäläinen (2002) — NDCG metric

---

## Target Publication Venues

- IEEE AI for Science workshops
- Springer Lecture Notes in Computer Science (AI/Astronomy)
- IJCAI / AAAI workshops on AI for Physical Sciences
- Monthly Notices of the Royal Astronomical Society (computational track)
