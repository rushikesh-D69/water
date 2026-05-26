"""
evaluation.py — Stage 2
=========================
Stage 2 Evaluation Metrics + Comparison Engine

Metrics:
  1. Cumulative Scientific Gain     — total knowledge acquired per round
  2. Telescope Utilization          — obs_time / total_time
  3. Regret@K                       — vs oracle scheduler
  4. Observation Efficiency         — Gain / Cost per round
  5. Uncertainty Reduction Rate     — delta_sigma / round
  6. Exploration Ratio              — unique targets / total budget

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


def build_comparison_table(results: Dict[str, dict], n_rounds: int, k_per_round: int) -> pd.DataFrame:
    """
    Build a summary comparison table across all schedulers.
    """
    rows = []
    max_gain = max(r["cumulative_gain"] for r in results.values())

    for name, res in results.items():
        logs = res["logs_df"]
        obs  = res["obs_history_df"]

        util_mean  = float(compute_telescope_utilization(logs).mean()) if not logs.empty else 0.0
        eff        = float(compute_observation_efficiency(logs).mean()) if not logs.empty else 0.0
        unc_rate   = compute_uncertainty_reduction_rate(obs)
        unc_mean   = float(unc_rate["mean_sigma_reduction"].mean()) if not unc_rate.empty else 0.0
        regret_fin = float(compute_regret(logs, max_gain).iloc[-1]) if not logs.empty else 1.0
        final_gain = res["cumulative_gain"]

        rows.append({
            "Scheduler":               name,
            "Cum. Sci. Gain":          round(final_gain, 4),
            "Regret (final)":          round(regret_fin, 4),
            "Telescope Utilization":   round(util_mean, 3),
            "Obs. Efficiency":         round(eff, 4),
            "Mean sigma Reduction":    round(unc_mean, 4),
            "Planets Observed":        res["n_observed"],
            "Total Hrs Used":          round(res["total_time_used"], 1),
        })

    df = pd.DataFrame(rows).sort_values("Cum. Sci. Gain", ascending=False).reset_index(drop=True)
    df.insert(0, "Rank", range(1, len(df) + 1))
    return df


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


def plot_regret(results: Dict[str, dict]):
    """Regret (vs best scheduler) per round."""
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor(DARK_BG)
    _style_ax(ax, "Regret vs Best Scheduler per Round", "Round", "Regret")

    max_gain = max(r["cumulative_gain"] for r in results.values())

    for name, res in results.items():
        logs   = res["logs_df"]
        color  = SCHEDULER_COLORS.get(name, TEXT)
        lw     = 2.5 if name == "Adaptive Scheduler" else 1.5
        if logs.empty or name == "Adaptive Scheduler":
            continue
        regret = compute_regret(logs, max_gain)
        ax.plot(logs["round"], regret, color=color, lw=lw, ls="--", label=name)

    ax.axhline(0, color=ACCENT, lw=2, label="Adaptive Scheduler (oracle reference)")
    ax.legend(facecolor=DARK_BG, edgecolor="#30363d", labelcolor=TEXT)
    ax.grid(axis="y", color="#30363d", alpha=0.4)

    plt.tight_layout()
    out = PLOTS_DIR / "s2_regret.png"
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
    results:     Dict[str, dict],
    n_rounds:    int = 30,
    k_per_round: int = 10,
    n_planets:   int = 5522,
    weather_history: List[float] = None,
) -> pd.DataFrame:
    """
    Run all evaluation metrics and generate all plots.

    Parameters
    ----------
    results      : dict  output from run_campaign() for each scheduler
    n_rounds     : int
    k_per_round  : int
    n_planets    : int   total planet pool size
    weather_history : list of weather values across rounds

    Returns
    -------
    comparison_df : pd.DataFrame  summary table
    """
    print("\n[Eval] Computing metrics ...")
    comparison_df = build_comparison_table(results, n_rounds, k_per_round)

    print("\n[Eval] Generating plots ...")
    plot_cumulative_gain(results)
    plot_uncertainty_evolution(results)
    plot_weight_decay(n_rounds)
    plot_regret(results)
    plot_observation_efficiency(results)
    if weather_history:
        plot_weather_sequence(weather_history)

    print("\n[Eval] Scheduler Comparison:")
    print(comparison_df.to_string(index=False))
    return comparison_df
