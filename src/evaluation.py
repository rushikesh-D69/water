"""
evaluation.py — Stage 2
=========================
Stage 2 Evaluation Metrics + Comparison Engine

Metrics:
  1. Cumulative Scientific Gain     — total knowledge acquired per round
  2. Telescope Utilization          — obs_time / total_time
  3. Regret@K vs Oracle             — absolute regret vs OracleScheduler upper bound
  4. Observation Efficiency         — Gain / Cost per round
  5. Uncertainty Reduction Rate     — delta_sigma / round
  6. Exploration Ratio              — unique targets / total budget
  7. Campaign Diversity Score       — parameter-space coverage across 5 dimensions
                                      (stellar type, temperature, orbital period,
                                       planet mass, system distance)

Also: comparison tables, convergence plots, scheduler ranking.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Dict, List

# ── Theme ─────────────────────────────────────────────────────────────────────
DARK_BG = "#0d1117"
PANEL   = "#161b22"
ACCENT  = "#22b5a0"
GOLD    = "#f0a500"
PINK    = "#c9ada7"
TEXT    = "#e6edf3"
MUTED   = "#8b949e"
BLUE    = "#7bccf6"

SCHEDULER_COLORS = {
    "Static Priority":           GOLD,
    "Detectability Greedy":      PINK,
    "Uncertainty Greedy":        BLUE,
    "Adaptive Scheduler":        ACCENT,
    "Oracle":                    "#ffffff",   # white — theoretical upper bound
}

PLOTS_DIR = Path(__file__).resolve().parent.parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


def _style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_edgecolor("#30363d")
    ax.tick_params(colors=MUTED)
    if title:  ax.set_title(title, color=TEXT, fontsize=11)
    if xlabel: ax.set_xlabel(xlabel, color=TEXT)
    if ylabel: ax.set_ylabel(ylabel, color=TEXT)


# =============================================================================
# 1. Metric Computations
# =============================================================================

def compute_telescope_utilization(logs_df: pd.DataFrame, total_hrs_per_round: float = 8.0) -> pd.Series:
    """Utilization = time_used / total_budget per round."""
    return (logs_df["time_used_hrs"] / total_hrs_per_round).clip(0, 1)


def compute_regret(
    logs_df:           pd.DataFrame,
    oracle_cum_gain:   float,
) -> pd.DataFrame:
    """
    Regret@round = (oracle_cum_gain_at_round - achieved_cum_gain) / oracle_cum_gain
    Oracle is approximated as the best-performing scheduler.
    """
    regret = (oracle_cum_gain - logs_df["cum_sci_gain"].clip(upper=oracle_cum_gain)) / (oracle_cum_gain + 1e-8)
    return regret.clip(0, 1)


def compute_observation_efficiency(logs_df: pd.DataFrame) -> pd.Series:
    """Efficiency = cumulative gain increment / time used per round."""
    gain_delta = logs_df["cum_sci_gain"].diff().fillna(logs_df["cum_sci_gain"].iloc[0])
    efficiency = gain_delta / (logs_df["time_used_hrs"] + 1e-6)
    return efficiency


def compute_uncertainty_reduction_rate(obs_history_df: pd.DataFrame) -> pd.DataFrame:
    """Mean sigma reduction per round."""
    if obs_history_df.empty:
        return pd.DataFrame()
    return (
        obs_history_df.groupby("round")["sigma_reduction"]
        .agg(["mean", "sum", "count"])
        .rename(columns={"mean": "mean_sigma_reduction", "sum": "total_sigma_reduced", "count": "n_observations"})
    )


def compute_exploration_ratio(logs_df: pd.DataFrame, n_planets: int) -> pd.Series:
    """
    Running ratio of unique planets scheduled / total scheduling budget used.
    Higher = more exploration.
    """
    cumulative_budget = np.arange(1, len(logs_df) + 1) * logs_df["n_selected"].mean()
    unique_per_round  = logs_df["n_selected"].cumsum()
    return (unique_per_round / (cumulative_budget + 1e-6)).clip(0, 1)


def compute_campaign_diversity(
    observed_indices: list,
    df:               "pd.DataFrame",
) -> dict:
    """
    Campaign Diversity Score: measures parameter-space coverage of selected planets.

    Computes diversity across 5 astrophysical dimensions:
      1. Stellar Type Entropy  : Shannon entropy of spectral class distribution
      2. Temperature Range     : std(T_eq) / max(T_eq) for observed planets
      3. Orbital Period Range  : std(log P_orb) / mean(log P_orb)
      4. Planet Mass Range     : std(log M_p) / mean(log M_p)
      5. Distance Coverage     : std(d_sys) / median(d_sys)

    Combined Diversity Score = mean of all 5 normalised diversity dimensions.

    Higher score = scheduler explored wider parameter space.
    Low score = scheduler clustered in one region (exploitation pathology).

    Parameters
    ----------
    observed_indices : list of int  — planet indices observed across full campaign
    df               : pd.DataFrame  — full processed planet dataframe

    Returns
    -------
    dict with individual dimension scores and combined diversity score
    """
    if not observed_indices:
        return {"diversity_score": 0.0}

    obs = df.iloc[list(observed_indices)].copy()

    scores = {}

    # 1. Stellar type entropy
    if "spectral_class" in obs.columns:
        counts  = obs["spectral_class"].value_counts(normalize=True)
        entropy = float(-np.sum(counts * np.log(counts + 1e-10)))
        max_ent = np.log(7.0)   # 7 spectral types
        scores["stellar_type_entropy"] = float(np.clip(entropy / max_ent, 0, 1))
    else:
        scores["stellar_type_entropy"] = 0.0

    # 2. Equilibrium temperature diversity
    teq = obs["pl_eqt"].dropna().values
    if len(teq) > 1:
        scores["temperature_diversity"] = float(
            np.clip(np.std(teq) / (np.max(teq) + 1e-6), 0, 1)
        )
    else:
        scores["temperature_diversity"] = 0.0

    # 3. Orbital period diversity (log scale)
    per = obs["pl_orbper"].dropna().values
    per = per[per > 0]
    if len(per) > 1:
        log_per = np.log1p(per)
        scores["orbital_diversity"] = float(
            np.clip(np.std(log_per) / (np.mean(log_per) + 1e-6), 0, 1)
        )
    else:
        scores["orbital_diversity"] = 0.0

    # 4. Planet mass diversity (log scale)
    mass = obs["pl_bmasse"].dropna().values
    mass = mass[mass > 0]
    if len(mass) > 1:
        log_mass = np.log1p(mass)
        scores["mass_diversity"] = float(
            np.clip(np.std(log_mass) / (np.mean(log_mass) + 1e-6), 0, 1)
        )
    else:
        scores["mass_diversity"] = 0.0

    # 5. System distance coverage
    dist = obs["sy_dist"].dropna().values
    dist = dist[dist > 0]
    if len(dist) > 1:
        scores["distance_coverage"] = float(
            np.clip(np.std(dist) / (np.median(dist) + 1e-6), 0, 1)
        )
    else:
        scores["distance_coverage"] = 0.0

    scores["diversity_score"] = float(np.mean(list(scores.values())))
    return scores


def build_comparison_table(
    results:   Dict[str, dict],
    n_rounds:  int,
    k_per_round: int,
    df:        "pd.DataFrame" = None,
    oracle_cum_gain: float = None,
) -> pd.DataFrame:
    """
    Build a summary comparison table across all schedulers.
    Uses OracleScheduler cumulative gain for regret if available.
    """
    rows = []
    # Use Oracle gain if provided, else use best scheduler gain
    if oracle_cum_gain is None:
        oracle_cum_gain = max(r["cumulative_gain"] for r in results.values())

    for name, res in results.items():
        logs = res["logs_df"]
        obs  = res["obs_history_df"]

        util_mean  = float(compute_telescope_utilization(logs).mean()) if not logs.empty else 0.0
        eff        = float(compute_observation_efficiency(logs).mean()) if not logs.empty else 0.0
        unc_rate   = compute_uncertainty_reduction_rate(obs)
        unc_mean   = float(unc_rate["mean_sigma_reduction"].mean()) if not unc_rate.empty else 0.0
        regret_fin = float(compute_regret(logs, oracle_cum_gain).iloc[-1]) if not logs.empty else 1.0
        final_gain = res["cumulative_gain"]

        # Diversity score
        if df is not None and not obs.empty and "planet_idx" in obs.columns:
            obs_idx = obs["planet_idx"].unique().tolist()
            div     = compute_campaign_diversity(obs_idx, df)
            div_score = round(div["diversity_score"], 4)
        else:
            div_score = 0.0

        rows.append({
            "Scheduler":               name,
            "Cum. Sci. Gain":          round(final_gain, 4),
            "Regret vs Oracle":        round(regret_fin, 4),
            "Diversity Score":         div_score,
            "Telescope Utilization":   round(util_mean, 3),
            "Obs. Efficiency":         round(eff, 4),
            "Mean sigma Reduction":    round(unc_mean, 4),
            "Planets Observed":        res["n_observed"],
            "Total Hrs Used":          round(res["total_time_used"], 1),
        })

    df_out = pd.DataFrame(rows).sort_values("Cum. Sci. Gain", ascending=False).reset_index(drop=True)
    df_out.insert(0, "Rank", range(1, len(df_out) + 1))
    return df_out


# =============================================================================
# 2. Plots
# =============================================================================

def plot_cumulative_gain(results: Dict[str, dict]):
    """Cumulative scientific gain per round — main comparison plot."""
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(DARK_BG)
    _style_ax(ax, "Cumulative Scientific Gain per Round", "Round", "Cumulative Scientific Gain")

    for name, res in results.items():
        logs = res["logs_df"]
        if logs.empty:
            continue
        color = SCHEDULER_COLORS.get(name, TEXT)
        lw    = 3 if name == "Adaptive Scheduler" else 1.5
        ls    = "-" if name == "Adaptive Scheduler" else "--"
        ax.plot(logs["round"], logs["cum_sci_gain"], color=color,
                lw=lw, ls=ls, label=name, alpha=0.9)
        # Final value annotation
        ax.annotate(f"{logs['cum_sci_gain'].iloc[-1]:.3f}",
                    xy=(logs["round"].iloc[-1], logs["cum_sci_gain"].iloc[-1]),
                    color=color, fontsize=9, ha="left", va="center")

    ax.legend(facecolor=DARK_BG, edgecolor="#30363d", labelcolor=TEXT)
    ax.grid(axis="y", color="#30363d", alpha=0.5)
    fig.suptitle("Stage 2: Adaptive vs Baseline Schedulers", color=TEXT, fontsize=13)
    plt.tight_layout()
    out = PLOTS_DIR / "s2_cumulative_gain.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"[Plot] Saved -> {out}")


def plot_uncertainty_evolution(results: Dict[str, dict]):
    """Mean prediction uncertainty over rounds per scheduler."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.patch.set_facecolor(DARK_BG)

    for name, res in results.items():
        logs  = res["logs_df"]
        color = SCHEDULER_COLORS.get(name, TEXT)
        lw    = 2.5 if name == "Adaptive Scheduler" else 1.5
        if logs.empty:
            continue
        axes[0].plot(logs["round"], logs["mean_sigma_before"],
                     color=color, lw=lw, label=name, alpha=0.85)
        axes[1].plot(logs["round"], logs["mean_priority"],
                     color=color, lw=lw, label=name, alpha=0.85)

    for ax, title, ylabel in zip(axes,
        ["Mean Prediction Uncertainty (sigma) per Round",
         "Mean Priority of Selected Targets per Round"],
        ["Mean sigma", "Mean Priority Score"]):
        _style_ax(ax, title, "Round", ylabel)
        ax.legend(facecolor=DARK_BG, edgecolor="#30363d", labelcolor=TEXT, fontsize=8)
        ax.grid(axis="y", color="#30363d", alpha=0.4)

    plt.tight_layout()
    out = PLOTS_DIR / "s2_uncertainty_evolution.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"[Plot] Saved -> {out}")


def plot_weight_decay(n_rounds: int = 30, beta_0: float = 0.30, tau: float = 15.0):
    """Visualise the exploration weight decay schedule for the adaptive scheduler."""
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(DARK_BG)
    _style_ax(ax, "Adaptive Scheduler: Exploration-Exploitation Weight Schedule",
              "Round", "Weight Value")

    rounds  = np.arange(1, n_rounds + 1)
    beta_t  = beta_0 * np.exp(-rounds / tau)
    alpha_t = np.full_like(rounds, 0.50, dtype=float)
    gamma   = np.full_like(rounds, 0.20, dtype=float)

    # Normalize
    total   = alpha_t + beta_t + gamma
    ax.plot(rounds, alpha_t / total, color=ACCENT, lw=2, label="alpha (uncertainty) — constant exploration")
    ax.plot(rounds, beta_t / total,  color=GOLD,   lw=2, label=f"beta_t (priority) — decays (tau={tau})")
    ax.plot(rounds, gamma / total,   color=PINK,   lw=2, label="gamma (detectability) — constant")
    ax.axvline(tau, color=MUTED, lw=1, ls=":", alpha=0.7, label=f"tau = {tau} rounds")

    ax.fill_between(rounds, 0, beta_t / total, alpha=0.08, color=GOLD)
    ax.legend(facecolor=DARK_BG, edgecolor="#30363d", labelcolor=TEXT)
    ax.grid(axis="y", color="#30363d", alpha=0.4)
    ax.set_ylim(0, 1.0)

    plt.tight_layout()
    out = PLOTS_DIR / "s2_weight_decay.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"[Plot] Saved -> {out}")


def plot_weather_sequence(weather_history: List[float]):
    """AR(1) weather quality across rounds."""
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor(DARK_BG)
    _style_ax(ax, "AR(1) Weather Quality Across Rounds (rho=0.65)", "Round", "Weather Quality")

    rounds = np.arange(1, len(weather_history) + 1)
    ax.fill_between(rounds, 0, weather_history, alpha=0.25, color=BLUE)
    ax.plot(rounds, weather_history, color=BLUE, lw=1.5)
    ax.axhline(0.65, color=MUTED, lw=1, ls="--", alpha=0.6, label="Fair weather threshold")
    ax.set_ylim(0, 1.05)
    ax.legend(facecolor=DARK_BG, edgecolor="#30363d", labelcolor=TEXT)

    plt.tight_layout()
    out = PLOTS_DIR / "s2_weather_sequence.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"[Plot] Saved -> {out}")


def plot_regret(results: Dict[str, dict], oracle_cum_gain: float = None):
    """Regret vs Oracle (or best scheduler) per round."""
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor(DARK_BG)

    if oracle_cum_gain is None:
        oracle_cum_gain = max(r["cumulative_gain"] for r in results.values())
        title = "Regret vs Best Scheduler per Round"
    else:
        title = "Regret vs Oracle (Perfect Knowledge) per Round"

    _style_ax(ax, title, "Round", "Regret")

    for name, res in results.items():
        logs   = res["logs_df"]
        color  = SCHEDULER_COLORS.get(name, TEXT)
        lw     = 2.5 if name == "Adaptive Scheduler" else 1.5
        if logs.empty or name == "Oracle":
            continue
        regret = compute_regret(logs, oracle_cum_gain)
        ls = "-" if name == "Adaptive Scheduler" else "--"
        ax.plot(logs["round"], regret, color=color, lw=lw, ls=ls, label=name)

    ax.axhline(0, color="#ffffff", lw=1.5, ls=":", alpha=0.6, label="Oracle (upper bound)")
    ax.legend(facecolor=DARK_BG, edgecolor="#30363d", labelcolor=TEXT)
    ax.grid(axis="y", color="#30363d", alpha=0.4)

    plt.tight_layout()
    out = PLOTS_DIR / "s2_regret.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"[Plot] Saved -> {out}")


def plot_diversity(diversity_scores: Dict[str, dict]):
    """
    Radar/bar chart of Campaign Diversity Scores across schedulers.
    Shows 5 dimensions: stellar type, temperature, orbital, mass, distance.
    """
    dims = ["stellar_type_entropy", "temperature_diversity",
            "orbital_diversity", "mass_diversity", "distance_coverage"]
    labels = ["Stellar Type\nEntropy", "Temperature\nDiversity",
              "Orbital\nDiversity", "Mass\nDiversity", "Distance\nCoverage"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor(DARK_BG)

    # Left: grouped bar chart
    ax = axes[0]
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
    ax.tick_params(colors=MUTED)

    x      = np.arange(len(dims))
    n_sched = len(diversity_scores)
    width   = 0.8 / n_sched

    for i, (name, scores) in enumerate(diversity_scores.items()):
        color  = SCHEDULER_COLORS.get(name, TEXT)
        vals   = [scores.get(d, 0.0) for d in dims]
        offset = (i - n_sched / 2 + 0.5) * width
        ax.bar(x + offset, vals, width * 0.9, label=name, color=color, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, color=TEXT, fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_title("Campaign Diversity: 5-Dimensional Coverage", color=TEXT, fontsize=11)
    ax.set_ylabel("Diversity Score [0,1]", color=TEXT)
    ax.legend(facecolor=DARK_BG, edgecolor="#30363d", labelcolor=TEXT, fontsize=8)
    ax.grid(axis="y", color="#30363d", alpha=0.3)

    # Right: overall diversity score bar
    ax2 = axes[1]
    ax2.set_facecolor(PANEL)
    for sp in ax2.spines.values(): sp.set_edgecolor("#30363d")
    ax2.tick_params(colors=MUTED)

    names  = list(diversity_scores.keys())
    totals = [diversity_scores[n].get("diversity_score", 0.0) for n in names]
    colors = [SCHEDULER_COLORS.get(n, TEXT) for n in names]
    bars   = ax2.barh(names, totals, color=colors, alpha=0.85)
    for bar, val in zip(bars, totals):
        ax2.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{val:.3f}", va="center", color=TEXT, fontsize=9)
    ax2.set_xlim(0, 1.1)
    ax2.set_title("Overall Campaign Diversity Score", color=TEXT, fontsize=11)
    ax2.set_xlabel("Diversity Score [0,1]\n(higher = wider parameter-space coverage)", color=TEXT)
    ax2.tick_params(colors=TEXT)
    ax2.grid(axis="x", color="#30363d", alpha=0.3)

    plt.tight_layout()
    out = PLOTS_DIR / "s2_diversity.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"[Plot] Saved -> {out}")


def plot_observation_efficiency(results: Dict[str, dict]):
    """Observation efficiency (gain per telescope hour) per round."""
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor(DARK_BG)
    _style_ax(ax, "Observation Efficiency (Gain / Telescope Hour) per Round",
              "Round", "Efficiency")

    for name, res in results.items():
        logs  = res["logs_df"]
        color = SCHEDULER_COLORS.get(name, TEXT)
        lw    = 2.5 if name == "Adaptive Scheduler" else 1.5
        ls    = "-" if name == "Adaptive Scheduler" else "--"
        if logs.empty:
            continue
        eff = compute_observation_efficiency(logs)
        ax.plot(logs["round"], eff.rolling(3, min_periods=1).mean(),
                color=color, lw=lw, ls=ls, label=name, alpha=0.9)

    ax.legend(facecolor=DARK_BG, edgecolor="#30363d", labelcolor=TEXT)
    ax.grid(axis="y", color="#30363d", alpha=0.4)

    plt.tight_layout()
    out = PLOTS_DIR / "s2_efficiency.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"[Plot] Saved -> {out}")


def run_full_evaluation(
    results:         Dict[str, dict],
    n_rounds:        int = 30,
    k_per_round:     int = 10,
    n_planets:       int = 5522,
    weather_history: List[float] = None,
    df:              "pd.DataFrame" = None,
    oracle_cum_gain: float = None,
) -> pd.DataFrame:
    """
    Run all evaluation metrics and generate all plots.

    Parameters
    ----------
    results          : dict  output from run_campaign() for each scheduler
    n_rounds         : int
    k_per_round      : int
    n_planets        : int   total planet pool size
    weather_history  : list  weather per round (for AR1 plot)
    df               : pd.DataFrame  planet dataframe (for diversity metric)
    oracle_cum_gain  : float  OracleScheduler cumulative gain (for regret)

    Returns
    -------
    comparison_df : pd.DataFrame  summary table
    """
    print("\n[Eval] Computing metrics ...")
    comparison_df = build_comparison_table(
        results, n_rounds, k_per_round,
        df=df, oracle_cum_gain=oracle_cum_gain
    )

    print("\n[Eval] Generating plots ...")
    plot_cumulative_gain(results)
    plot_uncertainty_evolution(results)
    plot_weight_decay(n_rounds)
    plot_regret(results, oracle_cum_gain=oracle_cum_gain)
    plot_observation_efficiency(results)
    if weather_history:
        plot_weather_sequence(weather_history)

    # Campaign diversity
    if df is not None:
        diversity_scores = {}
        for name, res in results.items():
            obs = res["obs_history_df"]
            if not obs.empty and "planet_idx" in obs.columns:
                obs_idx = obs["planet_idx"].unique().tolist()
                diversity_scores[name] = compute_campaign_diversity(obs_idx, df)
        if diversity_scores:
            plot_diversity(diversity_scores)

    print("\n[Eval] Scheduler Comparison:")
    print(comparison_df.to_string(index=False))
    return comparison_df
