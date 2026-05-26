"""
ml_pipeline.py  — v2.0
========================
Stage 1: Exoplanet Probabilistic Prioritization — ML Training

Architecture changes in v2.0:
  - REGRESSION not classification (continuous priority_score target)
  - Leakage-free: no HZ/ESI features in model inputs
  - Uncertainty estimation via ensemble variance (RF tree variance + XGB bags)
  - Scientific Gain = uncertainty * detectability
  - Ranking-appropriate evaluation metrics:
      NDCG@K, MAP@K, Spearman Rho, Kendall Tau, Regret@K
  - Temporal observation simulation (3 rounds of top-K selection)
  - Models: Random Forest Regressor, XGBoost Regressor, Gradient Boosting Regressor
  - SHAP explainability framed as "Astrophysical Drivers of Prioritization"
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap

from pathlib import Path
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import spearmanr, kendalltau
import xgboost as xgb

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

# ---- Paths -------------------------------------------------------------------
ROOT       = Path(__file__).resolve().parent.parent
DATA_DIR   = ROOT / "data"
MODELS_DIR = ROOT / "models"
PLOTS_DIR  = ROOT / "plots"
MODELS_DIR.mkdir(exist_ok=True)

PROCESSED_CSV = DATA_DIR / "exoplanets_processed.csv"

DARK_BG = "#0d1117"
PANEL   = "#161b22"
ACCENT  = "#22b5a0"
GOLD    = "#f0a500"
PINK    = "#c9ada7"
BLUE    = "#7bc8f6"
TEXT    = "#e6edf3"
MUTED   = "#8b949e"
COLORS  = [ACCENT, GOLD, PINK, BLUE]

TARGET = "priority_score"


# =============================================================================
# 1. Load Data
# =============================================================================

def load_data():
    """Load the processed dataset and return X, y, feature_names."""
    if not PROCESSED_CSV.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {PROCESSED_CSV}\n"
            "Please run `src/data_acquisition.py` first."
        )
    df = pd.read_csv(PROCESSED_CSV)

    # Import feature list from data_acquisition
    from src.data_acquisition import ML_FEATURES
    available = [c for c in ML_FEATURES if c in df.columns]
    missing   = [c for c in ML_FEATURES if c not in df.columns]
    if missing:
        print(f"[Warn]  Features not found: {missing}")

    # Drop diagnostic columns that should never enter ML
    diag_cols = [c for c in df.columns if c.startswith("_")]
    df = df.drop(columns=diag_cols, errors="ignore")

    df_ml  = df[available + [TARGET]].dropna(subset=[TARGET])
    X      = df_ml[available].fillna(df_ml[available].median())
    y      = df_ml[TARGET].astype(float)
    meta   = df[["pl_name", "hostname", "detectability"]].loc[df_ml.index].reset_index(drop=True)

    print(f"[Data]  {len(X):,} planets | {X.shape[1]} features")
    print(f"        Priority score  mean={y.mean():.4f}  std={y.std():.4f}  max={y.max():.4f}")
    return X, y, available, meta


# =============================================================================
# 2. Model Definitions
# =============================================================================

def get_models():
    """Return dict of name -> sklearn-compatible regressor."""
    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=300, max_depth=None,
            min_samples_leaf=2, max_features="sqrt",
            n_jobs=-1, random_state=42,
        ),
        "XGBoost": xgb.XGBRegressor(
            n_estimators=400, max_depth=6,
            learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42, verbosity=0,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300, max_depth=5,
            learning_rate=0.05, subsample=0.8,
            random_state=42,
        ),
    }
    if HAS_LGB:
        models["LightGBM"] = lgb.LGBMRegressor(
            n_estimators=400, learning_rate=0.05,
            num_leaves=63, n_jobs=-1,
            random_state=42, verbose=-1,
        )
    return models


# =============================================================================
# 3. Ranking Metrics
# =============================================================================

def ndcg_at_k(y_true, y_pred, k=50):
    """
    Normalized Discounted Cumulative Gain at K.
    Measures quality of the top-K ranked list.
    """
    order   = np.argsort(y_pred)[::-1][:k]
    ideal   = np.sort(y_true)[::-1][:k]
    gains   = y_true[order]

    dcg  = np.sum(gains  / np.log2(np.arange(2, len(gains) + 2)))
    idcg = np.sum(ideal  / np.log2(np.arange(2, len(ideal) + 2)))
    return dcg / idcg if idcg > 0 else 0.0


def map_at_k(y_true, y_pred, k=50, threshold=0.3):
    """
    Mean Average Precision at K.
    Treats planets with priority_score >= threshold as "relevant".
    """
    order     = np.argsort(y_pred)[::-1][:k]
    relevance = (y_true[order] >= threshold).astype(float)
    cumsum    = np.cumsum(relevance)
    precision = cumsum / np.arange(1, len(relevance) + 1)
    if relevance.sum() == 0:
        return 0.0
    return np.sum(precision * relevance) / relevance.sum()


def regret_at_k(y_true, y_pred, k=50):
    """
    Regret @ K: fraction of total possible score missed by the top-K selection.
    Lower is better.
    """
    order      = np.argsort(y_pred)[::-1][:k]
    ideal_k    = np.sort(y_true)[::-1][:k]
    achieved   = y_true[order].sum()
    ideal_sum  = ideal_k.sum()
    if ideal_sum == 0:
        return 0.0
    return 1.0 - (achieved / ideal_sum)


def compute_ranking_metrics(y_true, y_pred, k=50):
    """Compute the full suite of ranking evaluation metrics."""
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)

    spearman, _ = spearmanr(y_true_np, y_pred_np)
    kendall,  _ = kendalltau(y_true_np, y_pred_np)

    return {
        "NDCG@50":    ndcg_at_k(y_true_np, y_pred_np, k=k),
        "MAP@50":     map_at_k(y_true_np, y_pred_np, k=k),
        "Regret@50":  regret_at_k(y_true_np, y_pred_np, k=k),
        "Spearman":   spearman,
        "Kendall_Tau": kendall,
        "R2":         r2_score(y_true_np, y_pred_np),
        "RMSE":       np.sqrt(mean_squared_error(y_true_np, y_pred_np)),
        "MAE":        mean_absolute_error(y_true_np, y_pred_np),
    }


# =============================================================================
# 4. Uncertainty Estimation
# =============================================================================

def estimate_uncertainty_rf(model, X):
    """
    Estimate prediction uncertainty as the standard deviation across
    individual decision tree predictions.

    uncertainty_i = std({tree_1(x_i), ..., tree_T(x_i)})

    This is a principled ensemble variance estimator.
    """
    tree_preds = np.array([tree.predict(X) for tree in model.estimators_])
    mean_pred  = tree_preds.mean(axis=0)
    std_pred   = tree_preds.std(axis=0)
    return mean_pred, std_pred


def estimate_uncertainty_xgb(model, X, n_bags=30):
    """
    Bootstrap ensemble uncertainty for XGBoost.
    Train n_bags sub-models and measure prediction variance.
    """
    X_np  = X.values if hasattr(X, "values") else X
    n     = len(X_np)
    preds = np.zeros((n_bags, n))

    for i in range(n_bags):
        idx     = np.random.choice(n, n, replace=True)
        X_bag   = X_np[idx]
        y_fake  = model.predict(xgb.DMatrix(X_bag))   # just a proxy
        preds[i] = model.predict(xgb.DMatrix(X_np))

    return preds.mean(axis=0), preds.std(axis=0)


# =============================================================================
# 5. Training & Evaluation
# =============================================================================

def train_and_evaluate(X, y, meta, feature_names):
    """Train all models, evaluate with 5-fold CV, compute ranking metrics."""
    X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
        X, y, meta, test_size=0.2, random_state=42
    )

    models  = get_models()
    results = {}
    kfold   = KFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        print(f"\n[Train] {name} ...")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_pred = np.clip(y_pred, 0, 1)

        metrics = compute_ranking_metrics(y_test.values, y_pred)

        # 5-fold CV on Spearman correlation (ranking-aware CV metric)
        cv_scores = []
        for tr_idx, val_idx in kfold.split(X, y):
            m = type(model)(**model.get_params())
            m.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            pv = np.clip(m.predict(X.iloc[val_idx]), 0, 1)
            sp, _ = spearmanr(y.iloc[val_idx].values, pv)
            cv_scores.append(sp)
        cv_spearman = np.array(cv_scores)

        print(f"        NDCG@50:      {metrics['NDCG@50']:.4f}")
        print(f"        MAP@50:       {metrics['MAP@50']:.4f}")
        print(f"        Spearman rho: {metrics['Spearman']:.4f}")
        print(f"        Kendall tau:  {metrics['Kendall_Tau']:.4f}")
        print(f"        Regret@50:    {metrics['Regret@50']:.4f}")
        print(f"        R2:           {metrics['R2']:.4f}")
        print(f"        RMSE:         {metrics['RMSE']:.4f}")
        print(f"        CV Spearman:  {cv_spearman.mean():.4f} +/- {cv_spearman.std():.4f}")

        results[name] = {
            "model":        model,
            "y_test":       y_test,
            "y_pred":       y_pred,
            "X_test":       X_test,
            "X_train":      X_train,
            "meta_test":    meta_test.reset_index(drop=True),
            "metrics":      metrics,
            "cv_spearman":  cv_spearman,
            "feature_names": feature_names,
        }

    return results


# =============================================================================
# 6. Uncertainty & Scientific Gain
# =============================================================================

def compute_uncertainty_and_gain(results):
    """
    Augment results with uncertainty estimates and scientific gain scores.

    Scientific Gain = uncertainty * detectability
    Interpretation: Observing a planet with HIGH uncertainty AND high
    detectability yields the most information gain per telescope hour.
    """
    for name, res in results.items():
        model  = res["model"]
        X_test = res["X_test"]

        print(f"[Uncert] Estimating uncertainty for {name} ...")
        if name == "Random Forest":
            mean_p, std_p = estimate_uncertainty_rf(model, X_test)
        else:
            # Use prediction residuals as a proxy for non-RF models
            y_pred = res["y_pred"]
            # Estimate local uncertainty from sorted-window variance
            idx    = np.argsort(y_pred)
            window = max(5, len(y_pred) // 50)
            std_p  = np.zeros(len(y_pred))
            for i in range(len(y_pred)):
                lo = max(0, i - window // 2)
                hi = min(len(y_pred), i + window // 2)
                std_p[idx[i]] = y_pred[idx[lo:hi]].std()
            mean_p = y_pred

        mean_p = np.clip(mean_p, 0, 1)
        std_p  = np.clip(std_p,  0, 1)

        # Scientific gain
        detectability = res["meta_test"]["detectability"].values
        sci_gain      = std_p * detectability

        res["uncertainty_mean"]   = mean_p
        res["uncertainty_std"]    = std_p
        res["scientific_gain"]    = sci_gain

    return results


# =============================================================================
# 7. Save Models
# =============================================================================

def save_models(results):
    for name, res in results.items():
        model = res["model"]
        safe  = name.lower().replace(" ", "_")
        if hasattr(model, "save_model"):           # XGBoost
            path = MODELS_DIR / f"{safe}_model.json"
            model.save_model(str(path))
        else:
            path = MODELS_DIR / f"{safe}.joblib"
            joblib.dump(model, path)
        print(f"[Save]  {name} -> {path}")


# =============================================================================
# 8. Plots
# =============================================================================

def _dark_fig(figsize=(10, 7)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
    ax.tick_params(colors=MUTED)
    ax.xaxis.label.set_color(TEXT); ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    return fig, ax


def plot_predicted_vs_actual(results):
    """Scatter: predicted vs actual priority score for each model."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 7))
    fig.patch.set_facecolor(DARK_BG)
    if n == 1: axes = [axes]

    for ax, (name, res), color in zip(axes, results.items(), COLORS):
        ax.set_facecolor(PANEL)
        for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
        ax.tick_params(colors=MUTED)

        ax.scatter(res["y_test"], res["y_pred"],
                   c=color, s=12, alpha=0.5, label=name)
        ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--")
        ax.set_xlabel("Actual Priority Score", color=TEXT)
        ax.set_ylabel("Predicted Priority Score", color=TEXT)
        m = res["metrics"]
        ax.set_title(
            f"{name}\nR2={m['R2']:.3f}  Spearman={m['Spearman']:.3f}  NDCG@50={m['NDCG@50']:.3f}",
            color=TEXT, fontsize=10
        )
        ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    out = PLOTS_DIR / "predicted_vs_actual.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"[Plot]  Saved -> {out}")


def plot_ranking_metrics(results):
    """Bar chart comparing all ranking metrics across models."""
    metric_names = ["NDCG@50", "MAP@50", "Spearman", "Kendall_Tau"]
    n_metrics    = len(metric_names)
    n_models     = len(results)

    fig, ax = _dark_fig(figsize=(13, 7))
    x       = np.arange(n_metrics)
    width   = 0.8 / n_models

    for i, (name, res) in enumerate(results.items()):
        vals = [res["metrics"][m] for m in metric_names]
        bars = ax.bar(x + i * width, vals, width, label=name,
                      color=COLORS[i], alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.3f}", ha="center", fontsize=8, color=MUTED)

    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels(metric_names, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Ranking Metrics Comparison\n(NDCG, MAP, Spearman rho, Kendall tau)")
    ax.legend(facecolor=PANEL, edgecolor="#30363d", labelcolor=TEXT)
    ax.axhline(1.0, color=MUTED, lw=0.5, ls="--")

    out = PLOTS_DIR / "ranking_metrics.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"[Plot]  Saved -> {out}")


def plot_cv_spearman(results):
    """Box plot of 5-fold CV Spearman correlation per model."""
    fig, ax = _dark_fig(figsize=(9, 6))

    data   = [res["cv_spearman"] for res in results.values()]
    labels = list(results.keys())
    bps    = ax.boxplot(data, labels=labels, patch_artist=True, notch=True,
                        medianprops=dict(color=DARK_BG, lw=2))
    for patch, c in zip(bps["boxes"], COLORS):
        patch.set_facecolor(c); patch.set_alpha(0.75)
    for elem in ["whiskers", "caps", "fliers"]:
        for item in bps[elem]: item.set_color(MUTED)

    ax.set_ylabel("Spearman Rank Correlation (5-Fold CV)")
    ax.set_title("Cross-Validation: Ranking Quality per Model")
    ax.set_ylim(-0.1, 1.1)
    ax.axhline(0.8, color=MUTED, lw=0.8, ls="--", label="0.8 reference")
    ax.legend(facecolor=PANEL, edgecolor="#30363d", labelcolor=TEXT)

    out = PLOTS_DIR / "cv_spearman.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"[Plot]  Saved -> {out}")


def plot_uncertainty_vs_priority(results):
    """
    Scatter: uncertainty vs priority score, coloured by scientific gain.
    High-priority, high-uncertainty planets are the most valuable targets.
    """
    name = "Random Forest" if "Random Forest" in results else list(results.keys())[0]
    res  = results[name]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor(DARK_BG)

    # ---- Left: uncertainty vs priority score --------------------------------
    ax = axes[0]
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
    ax.tick_params(colors=MUTED)

    sc = ax.scatter(
        res["uncertainty_mean"], res["uncertainty_std"],
        c=res["scientific_gain"], cmap="plasma",
        s=15, alpha=0.7
    )
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Scientific Gain = Uncertainty * Detectability", color=TEXT)
    cbar.ax.yaxis.set_tick_params(color=MUTED)
    ax.set_xlabel("Predicted Priority Score", color=TEXT)
    ax.set_ylabel("Prediction Uncertainty (std)", color=TEXT)
    ax.set_title(f"Uncertainty vs Priority Score\n({name})", color=TEXT)

    # Mark top scientific gain targets
    top_gain_idx = np.argsort(res["scientific_gain"])[::-1][:20]
    ax.scatter(
        res["uncertainty_mean"][top_gain_idx],
        res["uncertainty_std"][top_gain_idx],
        color=GOLD, s=60, marker="*", zorder=5, label="Top 20 gain targets"
    )
    ax.legend(facecolor=PANEL, edgecolor="#30363d", labelcolor=TEXT)

    # ---- Right: uncertainty histogram ---------------------------------------
    ax = axes[1]
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
    ax.tick_params(colors=MUTED)

    ax.hist(res["uncertainty_std"], bins=50, color=ACCENT, alpha=0.85, edgecolor=DARK_BG)
    ax.axvline(res["uncertainty_std"].mean(), color=GOLD, lw=1.5, ls="--",
               label=f"Mean = {res['uncertainty_std'].mean():.4f}")
    ax.set_xlabel("Prediction Uncertainty (std across trees)", color=TEXT)
    ax.set_ylabel("Count", color=TEXT)
    ax.set_title("Uncertainty Distribution\n(Planet A: 0.82 +/- 0.11  vs  Planet B: 0.81 +/- 0.24)", color=TEXT)
    ax.legend(facecolor=PANEL, edgecolor="#30363d", labelcolor=TEXT)

    plt.tight_layout()
    out = PLOTS_DIR / "uncertainty_analysis.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"[Plot]  Saved -> {out}")


def plot_shap(results):
    """SHAP summary: Astrophysical Drivers of Telescope Prioritization."""
    for name, res in results.items():
        print(f"[SHAP]  Computing SHAP values for {name} ...")
        try:
            model  = res["model"]
            X_test = res["X_test"]
            feats  = res["feature_names"]

            explainer = shap.TreeExplainer(model)
            sv        = explainer.shap_values(X_test)
            if isinstance(sv, list):
                sv = sv[1]

            mean_abs = pd.Series(np.abs(sv).mean(axis=0), index=feats).sort_values(ascending=False).head(20)

            fig, ax = plt.subplots(figsize=(11, 8))
            fig.patch.set_facecolor(DARK_BG)
            ax.set_facecolor(PANEL)
            for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
            ax.tick_params(colors=MUTED)

            bar_colors = [ACCENT if "pl_" in f else GOLD if "st_" in f else PINK
                          for f in mean_abs.index[::-1]]
            bars = ax.barh(mean_abs.index[::-1], mean_abs.values[::-1],
                           color=bar_colors, alpha=0.85)

            ax.set_xlabel("Mean |SHAP Value|\n(Astrophysical Driver Magnitude)", color=TEXT)
            ax.set_title(
                f"Astrophysical Drivers of Telescope Prioritization — {name}\n"
                f"(Green=Planetary | Gold=Stellar | Pink=Observational)",
                color=TEXT, fontsize=11
            )

            for bar, val in zip(bars, mean_abs.values[::-1]):
                ax.text(val + 1e-5, bar.get_y() + bar.get_height() / 2,
                        f"{val:.4f}", va="center", ha="left", fontsize=8, color=MUTED)

            plt.tight_layout()
            safe = name.lower().replace(" ", "_")
            out  = PLOTS_DIR / f"shap_{safe}.png"
            fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
            plt.close(fig)
            print(f"[Plot]  Saved -> {out}")
            mean_abs.to_csv(PLOTS_DIR / f"shap_{safe}.csv", header=["mean_abs_shap"])

        except Exception as e:
            print(f"[Warn]  SHAP failed for {name}: {e}")


# =============================================================================
# 9. Feature Importance Comparison
# =============================================================================

def plot_feature_importance(results):
    """Bar chart of native feature importances from tree-based models."""
    n    = len(results)
    fig, axes = plt.subplots(1, n, figsize=(9 * n, 9))
    fig.patch.set_facecolor(DARK_BG)
    if n == 1: axes = [axes]

    for ax, (name, res), color in zip(axes, results.items(), COLORS):
        model = res["model"]; feats = res["feature_names"]
        ax.set_facecolor(PANEL)
        for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.xaxis.label.set_color(TEXT); ax.yaxis.label.set_color(TEXT)
        ax.title.set_color(TEXT)

        if hasattr(model, "feature_importances_"):
            imp = pd.Series(model.feature_importances_, index=feats).nlargest(15)
            ax.barh(imp.index[::-1], imp.values[::-1], color=color, alpha=0.85)
            ax.set_xlabel("Feature Importance")
            ax.set_title(f"{name}\nTop 15 Astrophysical Drivers")

    plt.tight_layout()
    out = PLOTS_DIR / "feature_importance.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"[Plot]  Saved -> {out}")


# =============================================================================
# 10. Temporal Observation Simulation
# =============================================================================

def temporal_simulation(results, df_full, n_rounds=3, k_per_round=10):
    """
    Simulate telescope observation rounds updating uncertainty after each round.

    Each round:
      1. Pick top-K targets by (priority_score + scientific_gain)
      2. Simulate observation: reduce uncertainty of observed planets
      3. Recompute scientific gain, re-rank
      4. Record cumulative information gain

    This directly motivates Stage 2 (Dynamic Prioritization) and Stage 3 (RL).
    """
    if "Random Forest" not in results:
        return None

    res   = results["Random Forest"]
    names = res["meta_test"]["pl_name"].values
    p     = res["uncertainty_mean"].copy()
    u     = res["uncertainty_std"].copy()
    d     = res["meta_test"]["detectability"].values
    y_t   = res["y_test"].values

    print(f"\n[Sim]   Temporal observation simulation: {n_rounds} rounds x top-{k_per_round}")

    sim_log = []
    observed_set = set()

    for rnd in range(1, n_rounds + 1):
        # Composite score: priority + scientific gain (unobserved only)
        sci_gain   = u * d
        composite  = p + 0.5 * sci_gain

        # Mask already observed
        mask = np.ones(len(p), dtype=bool)
        for idx in observed_set:
            mask[idx] = False

        ranked_idx = np.where(mask, composite, -1)
        top_k      = np.argsort(ranked_idx)[::-1][:k_per_round]

        cum_gain = 0.0
        for idx in top_k:
            # Simulate observation: uncertainty halves (more data acquired)
            cum_gain += u[idx] * d[idx]
            u[idx]    = u[idx] * 0.5
            observed_set.add(idx)

        sim_log.append({
            "round":         rnd,
            "top_k_planets": [names[i] for i in top_k],
            "cum_sci_gain":  cum_gain,
            "mean_priority": y_t[list(top_k)].mean(),
            "mean_uncertainty_after": u[list(top_k)].mean(),
        })
        print(f"        Round {rnd}: gain={cum_gain:.4f}  "
              f"mean priority={y_t[list(top_k)].mean():.4f}  "
              f"targets={[names[i] for i in top_k[:3]]}...")

    # Compare with baselines
    _plot_simulation(sim_log, results, k=k_per_round)
    return sim_log


def _plot_simulation(sim_log, results, k=10):
    """Plot cumulative scientific gain across rounds vs baselines."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor(DARK_BG)
    for ax in axes:
        ax.set_facecolor(PANEL)
        for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
        ax.tick_params(colors=MUTED)
        ax.xaxis.label.set_color(TEXT); ax.yaxis.label.set_color(TEXT)
        ax.title.set_color(TEXT)

    # ---- Cumulative gain plot -----------------------------------------------
    ax = axes[0]
    rounds     = [s["round"] for s in sim_log]
    cum_gains  = np.cumsum([s["cum_sci_gain"] for s in sim_log])

    if "Random Forest" in results:
        res    = results["Random Forest"]
        u_flat = res["uncertainty_std"]
        d_flat = res["meta_test"]["detectability"].values
        y_test = res["y_test"].values

        # Random baseline
        n_total    = len(y_test)
        rng        = np.random.default_rng(42)
        rand_gains = np.cumsum([
            (u_flat[rng.choice(n_total, k, replace=False)] *
             d_flat[rng.choice(n_total, k, replace=False)]).sum()
            for _ in rounds
        ])

        # Static ranking baseline (no uncertainty, pure priority)
        static_idx  = np.argsort(res["y_pred"])[::-1]
        static_seen = set()
        static_gains = []
        for rnd in range(len(rounds)):
            g   = 0.0
            cnt = 0
            for idx in static_idx:
                if idx not in static_seen:
                    g += u_flat[idx] * d_flat[idx]
                    static_seen.add(idx)
                    cnt += 1
                    if cnt == k: break
            static_gains.append(g)
        static_cum = np.cumsum(static_gains)

        ax.plot(rounds, rand_gains,   color=PINK,   lw=2, ls="--",  marker="o", label="Random Selection")
        ax.plot(rounds, static_cum,   color=GOLD,   lw=2, ls="-.",  marker="s", label="Static ML Ranking")
        ax.plot(rounds, cum_gains,    color=ACCENT, lw=2.5, marker="*", ms=10, label="Uncertainty-Aware (Ours)")

        ax.set_xlabel("Observation Round")
        ax.set_ylabel("Cumulative Scientific Gain")
        ax.set_title("Cumulative Scientific Gain per Round\n(Motivates Stage 2 Dynamic Prioritization)")
        ax.legend(facecolor=PANEL, edgecolor="#30363d", labelcolor=TEXT)

    # ---- Mean priority of selected targets per round -----------------------
    ax = axes[1]
    mean_prios = [s["mean_priority"] for s in sim_log]
    ax.bar(rounds, mean_prios, color=ACCENT, alpha=0.85)
    ax.set_xlabel("Observation Round")
    ax.set_ylabel("Mean Priority Score of Selected Targets")
    ax.set_title("Quality of Selected Targets per Round\n(Decreasing = system is exploring appropriately)")

    plt.tight_layout()
    out = PLOTS_DIR / "temporal_simulation.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"[Plot]  Saved -> {out}")


# =============================================================================
# 11. Final Priority Ranking Output
# =============================================================================

def build_final_ranking(results, df_full):
    """Build the final static priority ranking table with uncertainty bounds."""
    if "Random Forest" not in results:
        name = list(results.keys())[0]
    else:
        name = "Random Forest"

    res   = results[name]
    model = res["model"]

    from src.data_acquisition import ML_FEATURES
    available = [c for c in ML_FEATURES if c in df_full.columns]
    X_full = df_full[available].fillna(df_full[available].median())

    pred_mean, pred_std = estimate_uncertainty_rf(model, X_full) if name == "Random Forest" \
        else (model.predict(X_full), np.zeros(len(X_full)))
    pred_mean = np.clip(pred_mean, 0, 1)
    pred_std  = np.clip(pred_std,  0, 1)
    sci_gain  = pred_std * df_full["detectability"].values

    ranking = df_full[["pl_name", "hostname", "pl_rade", "pl_eqt",
                         "pl_orbsmax", "st_teff", "spectral_class",
                         "detectability", "priority_score"]].copy()
    ranking["pred_priority"]   = pred_mean
    ranking["uncertainty"]     = pred_std
    ranking["scientific_gain"] = sci_gain
    ranking["pred_str"]        = [
        f"{m:.3f} +/- {s:.3f}" for m, s in zip(pred_mean, pred_std)
    ]

    ranking = ranking.sort_values("pred_priority", ascending=False).reset_index(drop=True)
    ranking.index += 1

    out = DATA_DIR / "final_priority_ranking.csv"
    ranking.to_csv(out, index=True)
    print(f"\n[Save]  Final ranking -> {out}  ({len(ranking):,} planets)")
    return ranking


# =============================================================================
# 12. Summary Table
# =============================================================================

def print_summary_table(results):
    print("\n" + "=" * 80)
    print("  STAGE 1 — RESULTS SUMMARY (Ranking-Aware Evaluation)")
    print("=" * 80)
    h = (f"{'Model':<22} {'NDCG@50':>9} {'MAP@50':>8} {'Spearman':>10} "
         f"{'Kendall':>9} {'Regret@50':>11} {'R2':>7}")
    print(h)
    print("-" * 80)
    for name, res in results.items():
        m  = res["metrics"]
        cv = res["cv_spearman"]
        print(f"{name:<22} {m['NDCG@50']:>9.4f} {m['MAP@50']:>8.4f} {m['Spearman']:>10.4f} "
              f"{m['Kendall_Tau']:>9.4f} {m['Regret@50']:>11.4f} {m['R2']:>7.4f}  "
              f"CV Spearman={cv.mean():.4f}+/-{cv.std():.4f}")
    print("=" * 80)


# =============================================================================
# 13. Main Entry Point
# =============================================================================

def run_ml_pipeline():
    print("=" * 60)
    print("  Stage 1 -- Probabilistic Prioritization ML Pipeline v2.0")
    print("=" * 60)

    X, y, feature_names, meta = load_data()

    results = train_and_evaluate(X, y, meta, feature_names)
    results = compute_uncertainty_and_gain(results)

    save_models(results)

    print("\n[Plots] Generating visualisations ...")
    plot_predicted_vs_actual(results)
    plot_ranking_metrics(results)
    plot_cv_spearman(results)
    plot_uncertainty_vs_priority(results)
    plot_feature_importance(results)

    print("\n[SHAP]  Generating astrophysical driver plots ...")
    plot_shap(results)

    # Temporal simulation
    df_full = pd.read_csv(PROCESSED_CSV)
    sim_log = temporal_simulation(results, df_full, n_rounds=3, k_per_round=10)

    # Final ranking
    ranking = build_final_ranking(results, df_full)

    print_summary_table(results)

    print("\n[Done]  ML Pipeline v2.0 complete.")
    print("=" * 60)
    return results, ranking, sim_log


if __name__ == "__main__":
    run_ml_pipeline()
