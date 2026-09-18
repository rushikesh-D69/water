"""
evaluation.py — Stage 2
=========================
Stage 2 Evaluation Metrics + Comparison Engine

Metrics:
  1. Cumulative Scientific Gain     — total knowledge acquired per round
  2. Telescope Utilization          — obs_time / total_time
  3. Regret@K vs Oracle             — absolute regret vs perfect-knowledge Oracle
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

from src.plot_style import (
    PAPER_BG, PANEL, ACCENT, GOLD, PINK, TEXT, MUTED, BLUE, SPINE, GRID, DPI,
    SCHEDULER_COLORS, SCHEDULER_STYLES, style_ax as _style_ax, legend, savefig,
    export_ieee_figures,
)

PLOTS_DIR = Path(__file__).resolve().parent.parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


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
    Uses redefined Oracle Scheduler relative normalization for metrics
    to compute a robust Multi-Objective Composite Campaign Score.
    """
    # ── 1. Extract Oracle raw values as base for normalization ────────────────
    oracle_res = results.get("Oracle")
    if oracle_res is not None:
        oracle_gain = oracle_res["cumulative_gain"]
        oracle_logs = oracle_res["logs_df"]
        oracle_obs  = oracle_res["obs_history_df"]
        oracle_eff  = float(compute_observation_efficiency(oracle_logs).mean()) if not oracle_logs.empty else 1e-8
        if df is not None and not oracle_obs.empty and "planet_idx" in oracle_obs.columns:
            oracle_obs_idx = oracle_obs["planet_idx"].unique().tolist()
            oracle_div = compute_campaign_diversity(oracle_obs_idx, df)["diversity_score"]
            true_prio_col = "priority_score" if "priority_score" in df.columns else "priority_score"
            oracle_prio = float(df.iloc[oracle_obs_idx][true_prio_col].mean()) if true_prio_col in df.columns else 1e-8
        else:
            oracle_div  = 1.0
            oracle_prio = 1.0
    else:
        # Fallback if Oracle is not in results
        oracle_gain = max(r["cumulative_gain"] for r in results.values())
        oracle_eff  = max(float(compute_observation_efficiency(r["logs_df"]).mean()) for r in results.values() if not r["logs_df"].empty)
        oracle_div  = 0.6
        oracle_prio = 0.7

    # Ensure oracle_cum_gain uses Oracle's actual gain for regret clipping
    if oracle_cum_gain is None:
        oracle_cum_gain = oracle_gain

    rows = []
    for name, res in results.items():
        logs = res["logs_df"]
        obs  = res["obs_history_df"]

        util_mean  = float(compute_telescope_utilization(logs).mean()) if not logs.empty else 0.0
        eff        = float(compute_observation_efficiency(logs).mean()) if not logs.empty else 0.0
        unc_rate   = compute_uncertainty_reduction_rate(obs)
        unc_mean   = float(unc_rate["mean_sigma_reduction"].mean()) if not unc_rate.empty else 0.0
        regret_fin = float(compute_regret(logs, oracle_cum_gain).iloc[-1]) if not logs.empty else 1.0
        final_gain = res["cumulative_gain"]

        # ── 2. Campaign Diversity ─────────────────────────────────────────────
        if df is not None and not obs.empty and "planet_idx" in obs.columns:
            obs_idx = obs["planet_idx"].unique().tolist()
            div     = compute_campaign_diversity(obs_idx, df)
            div_score = div["diversity_score"]
        else:
            div_score = 0.0

        # ── 3. Priority Coverage ──────────────────────────────────────────────
        if df is not None and not obs.empty and "planet_idx" in obs.columns:
            obs_idx = obs["planet_idx"].unique().tolist()
            true_prio_col = "priority_score" if "priority_score" in df.columns else "priority_score"
            prio_cov = float(df.iloc[obs_idx][true_prio_col].mean()) if true_prio_col in df.columns else 0.0
        else:
            prio_cov = 0.0

        # ── 4. Unified Oracle-Relative Normalization ──────────────────────────
        g_norm = np.clip(final_gain / (oracle_gain + 1e-8), 0.0, 1.0)
        d_norm = np.clip(div_score / (oracle_div + 1e-8), 0.0, 1.0)
        e_norm = np.clip(eff / (oracle_eff + 1e-8), 0.0, 1.0)
        p_norm = np.clip(prio_cov / (oracle_prio + 1e-8), 0.0, 1.0)

        # ── 5. Composite Score Calculation ────────────────────────────────────
        comp_score = 0.35 * g_norm + 0.25 * d_norm + 0.20 * e_norm + 0.20 * p_norm

        rows.append({
            "Scheduler":               name,
            "Composite Score":         round(comp_score, 4),
            "Cum. Sci. Gain":          round(final_gain, 4),
            "Regret vs Oracle":        round(regret_fin, 4),
            "Diversity Score":         round(div_score, 4),
            "Priority Coverage":       round(prio_cov, 4),
            "Telescope Utilization":   round(util_mean, 3),
            "Obs. Efficiency":         round(eff, 4),
            "Mean sigma Reduction":    round(unc_mean, 4),
            "Planets Observed":        res["n_observed"],
            "Total Hrs Used":          round(res["total_time_used"], 1),
        })

    df_out = pd.DataFrame(rows).sort_values("Composite Score", ascending=False).reset_index(drop=True)
    df_out.insert(0, "Rank", range(1, len(df_out) + 1))
    return df_out



# =============================================================================
# 2. Plots
# =============================================================================

def plot_cumulative_gain(results: Dict[str, dict]):
    """Cumulative scientific gain per round — main comparison plot."""
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    _style_ax(ax, "Cumulative scientific gain per round", "Round", "Cumulative scientific gain")

    for name, res in results.items():
        logs = res["logs_df"]
        if logs.empty:
            continue
        st = SCHEDULER_STYLES.get(name, dict(color=TEXT, ls="-", marker="o", lw=1.5, ms=4))
        ax.plot(logs["round"], logs["cum_sci_gain"], color=st["color"],
                lw=st["lw"], ls=st["ls"], marker=st["marker"], ms=st["ms"],
                markevery=4, label=name)
        ax.annotate(f"{logs['cum_sci_gain'].iloc[-1]:.3f}",
                    xy=(logs["round"].iloc[-1], logs["cum_sci_gain"].iloc[-1]),
                    xytext=(6, {"Adaptive Scheduler": 8, "Oracle": -10}.get(name, 0)),
                    textcoords="offset points",
                    color=st["color"], fontsize=8, ha="left", va="center")

    legend(ax)
    fig.tight_layout()
    savefig(fig, PLOTS_DIR / "s2_cumulative_gain.png")


def plot_uncertainty_evolution(results: Dict[str, dict]):
    """Mean prediction uncertainty over rounds per scheduler."""
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))

    for name, res in results.items():
        logs = res["logs_df"]
        if logs.empty:
            continue
        st = SCHEDULER_STYLES.get(name, dict(color=TEXT, ls="-", marker="o", lw=1.5, ms=4))
        axes[0].plot(logs["round"], logs["mean_sigma_before"],
                     color=st["color"], lw=st["lw"], ls=st["ls"],
                     marker=st["marker"], ms=st["ms"], markevery=4, label=name)
        axes[1].plot(logs["round"], logs["mean_priority"],
                     color=st["color"], lw=st["lw"], ls=st["ls"],
                     marker=st["marker"], ms=st["ms"], markevery=4, label=name)

    for ax, title, ylabel in zip(axes,
        ["Mean prediction uncertainty per round",
         "Mean priority of selected targets"],
        [r"Mean $\sigma$", "Mean priority score"]):
        _style_ax(ax, title, "Round", ylabel)
        legend(ax, fontsize=7)

    fig.tight_layout()
    savefig(fig, PLOTS_DIR / "s2_uncertainty_evolution.png")


def plot_weight_decay(n_rounds: int = 30, beta_0: float = 0.30, tau: float = 15.0):
    """Visualise the exploration weight decay schedule for the adaptive scheduler."""
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    _style_ax(ax, "Adaptive weight schedule (renormalized)", "Round", "Weight")

    rounds  = np.arange(1, n_rounds + 1)
    beta_t  = beta_0 * np.exp(-rounds / tau)
    alpha_t = np.full_like(rounds, 0.50, dtype=float)
    gamma   = np.full_like(rounds, 0.20, dtype=float)
    total   = alpha_t + beta_t + gamma
    ax.plot(rounds, alpha_t / total, color=BLUE,  lw=2, ls="-",  label=r"$\alpha_t$ (uncertainty)")
    ax.plot(rounds, beta_t / total,  color=GOLD,  lw=2, ls="--", label=fr"$\beta_t$ (priority, $\tau={tau:g}$)")
    ax.plot(rounds, gamma / total,   color=PINK, lw=2, ls="-.", label=r"$\gamma_t$ (detectability)")
    ax.axvline(tau, color=MUTED, lw=1, ls=":", label=fr"$\tau={tau:g}$")
    ax.fill_between(rounds, 0, beta_t / total, alpha=0.12, color=GOLD)
    legend(ax, loc="upper left", ncol=2, fontsize=8)
    ax.set_ylim(0, 1.15)
    fig.tight_layout()
    savefig(fig, PLOTS_DIR / "s2_weight_decay.png")


def plot_weather_sequence(weather_history: List[float]):
    """AR(1) weather quality across rounds."""
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    _style_ax(ax, r"AR(1) weather quality ($\rho=0.65$)", "Round", "Weather quality")

    rounds = np.arange(1, len(weather_history) + 1)
    ax.fill_between(rounds, 0, weather_history, alpha=0.18, color=BLUE)
    ax.plot(rounds, weather_history, color=BLUE, lw=1.6)
    ax.axhline(0.65, color=MUTED, lw=1, ls="--", label="Fair-weather threshold")
    ax.set_ylim(0, 1.05)
    legend(ax)
    fig.tight_layout()
    savefig(fig, PLOTS_DIR / "s2_weather_sequence.png")


def plot_regret(results: Dict[str, dict], oracle_cum_gain: float = None):
    """Regret vs Oracle (or best scheduler) per round."""
    fig, ax = plt.subplots(figsize=(7.2, 3.4))

    if oracle_cum_gain is None:
        oracle_cum_gain = max(r["cumulative_gain"] for r in results.values())
        title = "Regret vs best scheduler per round"
    else:
        title = "Regret vs Oracle per round"

    _style_ax(ax, title, "Round", "Regret")

    for name, res in results.items():
        logs = res["logs_df"]
        if logs.empty or name == "Oracle":
            continue
        st = SCHEDULER_STYLES.get(name, dict(color=TEXT, ls="-", marker="o", lw=1.5, ms=4))
        regret = compute_regret(logs, oracle_cum_gain)
        ax.plot(logs["round"], regret, color=st["color"], lw=st["lw"], ls=st["ls"],
                marker=st["marker"], ms=st["ms"], markevery=4, label=name)

    ax.axhline(0, color=TEXT, lw=1.0, ls=":", label="Oracle (perfect-knowledge ref.)")
    legend(ax)
    fig.tight_layout()
    savefig(fig, PLOTS_DIR / "s2_regret.png")


def plot_diversity(diversity_scores: Dict[str, dict]):
    """
    Bar chart of Campaign Diversity Scores across schedulers.
    Shows 5 dimensions: stellar type, temperature, orbital, mass, distance.
    """
    dims = ["stellar_type_entropy", "temperature_diversity",
            "orbital_diversity", "mass_diversity", "distance_coverage"]
    labels = ["Stellar type", "Temperature", "Orbital", "Mass", "Distance"]

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))

    ax = axes[0]
    _style_ax(ax, "Campaign diversity by dimension", "", "Diversity [0, 1]")
    x = np.arange(len(dims))
    n_sched = len(diversity_scores)
    width = 0.8 / max(n_sched, 1)

    for i, (name, scores) in enumerate(diversity_scores.items()):
        color = SCHEDULER_COLORS.get(name, TEXT)
        vals = [scores.get(d, 0.0) for d in dims]
        offset = (i - n_sched / 2 + 0.5) * width
        ax.bar(x + offset, vals, width * 0.9, label=name, color=color, edgecolor=SPINE, linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1.0)
    legend(ax, fontsize=7)

    ax2 = axes[1]
    _style_ax(ax2, "Overall campaign diversity", "Diversity [0, 1]", "")
    names = list(diversity_scores.keys())
    totals = [diversity_scores[n].get("diversity_score", 0.0) for n in names]
    colors = [SCHEDULER_COLORS.get(n, TEXT) for n in names]
    bars = ax2.barh(names, totals, color=colors, edgecolor=SPINE, linewidth=0.4)
    for bar, val in zip(bars, totals):
        ax2.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{val:.3f}", va="center", color=TEXT, fontsize=8)
    ax2.set_xlim(0, 1.15)

    fig.tight_layout()
    savefig(fig, PLOTS_DIR / "s2_diversity.png")


def plot_observation_efficiency(results: Dict[str, dict]):
    """Observation efficiency (gain per telescope hour) per round."""
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    _style_ax(ax, "Observation efficiency (gain / hour)", "Round", "Efficiency")

    for name, res in results.items():
        logs = res["logs_df"]
        if logs.empty:
            continue
        st = SCHEDULER_STYLES.get(name, dict(color=TEXT, ls="-", marker="o", lw=1.5, ms=4))
        eff = compute_observation_efficiency(logs)
        ax.plot(logs["round"], eff.rolling(3, min_periods=1).mean(),
                color=st["color"], lw=st["lw"], ls=st["ls"],
                marker=st["marker"], ms=st["ms"], markevery=4, label=name)

    legend(ax)
    fig.tight_layout()
    savefig(fig, PLOTS_DIR / "s2_efficiency.png")


def plot_pareto_frontier(results: Dict[str, dict], df: "pd.DataFrame"):
    """
    Generate a 2D Pareto Frontier plot showing Gain vs Diversity.
    Marker size represents observation efficiency.
    """
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    _style_ax(ax, "Gain vs diversity", "Cumulative scientific gain", "Campaign diversity")

    points = []
    for name, res in results.items():
        gain = res["cumulative_gain"]
        obs  = res["obs_history_df"]
        logs = res["logs_df"]
        eff  = float(compute_observation_efficiency(logs).mean()) if not logs.empty else float(res.get("_eff", 0.0))
        if "_diversity" in res:
            div = float(res["_diversity"])
        elif df is not None and not obs.empty and "planet_idx" in obs.columns:
            obs_idx = obs["planet_idx"].unique().tolist()
            div = compute_campaign_diversity(obs_idx, df)["diversity_score"]
        else:
            div = 0.0
        points.append((name, gain, div, eff))

    frontier = []
    for name, g, d, eff in points:
        dominated = False
        for oname, og, od, oeff in points:
            if oname == name:
                continue
            if og >= g and od >= d and (og > g or od > d):
                dominated = True
                break
        if not dominated:
            frontier.append((name, g, d, eff))
    frontier = sorted(frontier, key=lambda x: x[1])

    for name, g, d, eff in points:
        st = SCHEDULER_STYLES.get(name, dict(color=TEXT, marker="o", ms=6))
        size = 80 + eff * 1600
        ax.scatter(g, d, color=st["color"], s=size, marker=st.get("marker", "o"),
                   edgecolors=SPINE, linewidths=0.6, zorder=3, label=name)

    if len(frontier) > 1:
        fx = [f[1] for f in frontier]
        fy = [f[2] for f in frontier]
        ax.plot(fx, fy, color=MUTED, linestyle="--", linewidth=1.2, zorder=2)

    legend(ax, fontsize=7, loc="lower right")
    gains = [p[1] for p in points]
    divs  = [p[2] for p in points]
    ax.set_xlim(min(gains) * 0.92, max(gains) * 1.08)
    ax.set_ylim(min(divs) * 0.92, min(1.0, max(divs) * 1.08))
    fig.tight_layout()
    savefig(fig, PLOTS_DIR / "s2_pareto_frontier.png")



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
            plot_pareto_frontier(results, df)


    print("\n[Eval] Scheduler Comparison:")
    print(comparison_df.to_string(index=False))
    export_ieee_figures()
    return comparison_df


def regenerate_paper_plots_from_logs(data_dir=None):
    """Rebuild IEEE-print Stage 2 figures from saved campaign CSVs."""
    data_dir = Path(data_dir or Path(__file__).resolve().parent.parent / "data")
    name_files = {
        "Adaptive Scheduler":   "s2_adaptive_scheduler_logs.csv",
        "Detectability Greedy": "s2_detectability_greedy_logs.csv",
        "Static Priority":      "s2_static_priority_logs.csv",
        "Uncertainty Greedy":   "s2_uncertainty_greedy_logs.csv",
        "Oracle":               "s2_oracle_logs.csv",
    }
    results = {}
    for name, fn in name_files.items():
        p = data_dir / fn
        if not p.exists():
            print(f"[Eval] missing {p}")
            continue
        logs = pd.read_csv(p)
        results[name] = {
            "logs_df": logs,
            "cumulative_gain": float(logs["cum_sci_gain"].iloc[-1]),
            "obs_history_df": pd.DataFrame(),
        }
    comp = data_dir / "stage2_comparison.csv"
    if comp.exists():
        cdf = pd.read_csv(comp)
        for _, row in cdf.iterrows():
            name = row["Scheduler"]
            if name in results:
                results[name]["_diversity"] = float(row["Diversity Score"])
                results[name]["_eff"] = float(row["Obs. Efficiency"])
                results[name]["cumulative_gain"] = float(row["Cum. Sci. Gain"])
    if not results:
        print("[Eval] no Stage 2 logs found")
        return
    plot_cumulative_gain(results)
    plot_uncertainty_evolution(results)
    plot_weight_decay()
    adapt = results.get("Adaptive Scheduler", {}).get("logs_df")
    if adapt is not None and "weather" in adapt.columns:
        plot_weather_sequence(adapt["weather"].tolist())
    plot_regret(results, oracle_cum_gain=results.get("Oracle", {}).get("cumulative_gain"))
    plot_observation_efficiency(results)
    plot_pareto_frontier(results, df=None)

    proc = data_dir / "exoplanets_processed.csv"
    if proc.exists():
        from src.data_acquisition import plot_feature_correlations, ML_FEATURES
        df = pd.read_csv(proc)
        plot_feature_correlations(df, ML_FEATURES)
    export_ieee_figures(names=[
        "s2_weather_sequence.png",
        "s2_weight_decay.png",
        "s2_cumulative_gain.png",
        "s2_pareto_frontier.png",
    ])


if __name__ == "__main__":
    regenerate_paper_plots_from_logs()
