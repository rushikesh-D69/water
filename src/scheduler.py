"""
scheduler.py — Stage 2
========================
Scheduler Algorithms + Adaptive Reprioritization Loop

Implements 5 schedulers sharing a common interface:
  1. StaticPriorityScheduler       — Baseline 1: always top priority_score
  2. DetectabilityGreedyScheduler  — Baseline 2: always top detectability
  3. UncertaintyGreedyScheduler    — Baseline 3: always highest uncertainty
  4. AdaptiveScheduler             — OUR METHOD: Utility = (Gain × F) / Cost
  5. OracleScheduler               — UPPER BOUND: perfect future knowledge

OracleScheduler (academic upgrade):
  Has access to the TRUE priority scores (ground truth labels).
  Selects the globally optimal set every round given constraints.
  Used exclusively to compute regret of all other schedulers:
    Regret@round = (Oracle_cum_gain - Scheduler_cum_gain) / Oracle_cum_gain
  This makes the regret metric rigorously meaningful.

Scientific Gain formula (upgraded, Issue 3):
  Gain_i = α_t × U_i + β_t × P_i + γ × D_i

  With time-decaying exploration weight (Issue 3):
    β_t = β_0 × exp(-t / tau)   ← early exploration → late exploitation

Utility function:
  Utility_i = (Gain_i × Feasibility_i) / Cost_i

All schedulers:
  - Respect telescope time budget per round
  - Skip planets already observed (unless budget allows re-obs)
  - Log per-round selection with full metadata
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

from src.constraint_engine import ObservationConstraintEngine
from src.observation_simulator import ObservationSimulator


# ── Round log entry ───────────────────────────────────────────────────────────
@dataclass
class RoundLog:
    round_number:       int
    scheduler_name:     str
    selected_indices:   List[int]
    selected_names:     List[str]
    utilities:          List[float]
    gains:              List[float]
    feasibilities:      List[float]
    costs:              List[float]
    weather:            float
    time_used:          float
    time_budget:        float
    alpha_t:            float
    beta_t:             float
    gamma:              float
    cum_sci_gain:       float
    mean_priority:      float
    mean_sigma_before:  float
    mean_sigma_after:   float


# ── Base scheduler interface ──────────────────────────────────────────────────
class BaseScheduler(ABC):
    """
    Abstract base for all schedulers.

    All schedulers implement:
      select(simulator, constraint_engine, round_number, k)
        → List[int] of planet indices to observe this round
    """

    def __init__(self, name: str, df: pd.DataFrame):
        self.name  = name
        self.df    = df.reset_index(drop=True)
        self.logs: List[RoundLog] = []
        self.cumulative_gain = 0.0
        self.total_time_used = 0.0

    @abstractmethod
    def select(
        self,
        simulator:         ObservationSimulator,
        constraint_engine: ObservationConstraintEngine,
        round_number:      int,
        k:                 int,
        observed_set:      set,
    ) -> Tuple[List[int], List[float]]:
        """
        Returns
        -------
        selected_indices : List[int]
        costs_per_planet : List[float]
        """

    def _feasible_candidates(
        self,
        constraint_engine: ObservationConstraintEngine,
        observed_set:      set,
        allow_reobs:       bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns indices, feasibilities, costs for all observable planets
        not yet observed (or all if allow_reobs=True).
        """
        all_idx = np.arange(constraint_engine.n_planets)
        if not allow_reobs:
            mask    = np.array([i not in observed_set for i in all_idx])
            all_idx = all_idx[mask]

        # Only planets currently visible
        vis_mask = constraint_engine.visibility[all_idx] > 0
        all_idx  = all_idx[vis_mask]

        if len(all_idx) == 0:
            return np.array([]), np.array([]), np.array([])

        feasibility, costs = constraint_engine.compute_feasibility(all_idx)
        return all_idx, feasibility, costs

    def _apply_budget(
        self,
        ranked_indices:    np.ndarray,
        costs:             np.ndarray,
        constraint_engine: ObservationConstraintEngine,
        k:                 int,
    ) -> Tuple[List[int], List[float]]:
        """
        Select top-k from ranked_indices, respecting time budget.
        Returns selected indices and their costs.
        """
        selected = []
        sel_costs = []
        n_selected = 0
        for rank, (idx, cost) in enumerate(zip(ranked_indices, costs)):
            can, total_cost = constraint_engine.can_observe(int(idx), n_selected)
            if not can:
                continue
            constraint_engine.consume_budget(int(idx), n_selected)
            selected.append(int(idx))
            sel_costs.append(total_cost)
            n_selected += 1
            if n_selected >= k:
                break
        return selected, sel_costs

    def _build_log(
        self,
        round_number:      int,
        selected:          List[int],
        sel_costs:         List[float],
        simulator:         ObservationSimulator,
        constraint_engine: ObservationConstraintEngine,
        gains_arr:         np.ndarray,
        feas_arr:          np.ndarray,
        cand_idx:          np.ndarray,
        alpha_t:           float,
        beta_t:            float,
        gamma:             float,
        time_before:       float,
    ) -> RoundLog:
        names = [
            str(self.df["pl_name"].iloc[i]) if "pl_name" in self.df.columns else f"P{i}"
            for i in selected
        ]

        # Map selected indices to gain/feas arrays
        idx_map = {int(i): pos for pos, i in enumerate(cand_idx)}
        sel_gains = [float(gains_arr[idx_map[i]]) if i in idx_map else 0.0 for i in selected]
        sel_feas  = [float(feas_arr[idx_map[i]])  if i in idx_map else 0.0 for i in selected]
        utilities = [
            g * f / (c + 1e-6) for g, f, c in zip(sel_gains, sel_feas, sel_costs)
        ]

        # ── Use numpy array indexing throughout — avoids list/Series ambiguity ──
        if selected:
            sel_idx       = np.array(selected, dtype=np.intp)
            sig_sel       = simulator.sigma[sel_idx]
            det_sel       = simulator.detectability[sel_idx]
            mu_sel        = simulator.mu[sel_idx]
            round_gain    = float(np.dot(sig_sel, det_sel))
            sigma_before  = float(np.mean(sig_sel))
            sigma_after   = sigma_before
            mean_priority = float(np.mean(mu_sel))
        else:
            round_gain    = 0.0
            sigma_before  = 0.0
            sigma_after   = 0.0
            mean_priority = 0.0

        self.cumulative_gain += round_gain
        time_used = time_before - constraint_engine.time_budget
        self.total_time_used += time_used

        return RoundLog(
            round_number      = round_number,
            scheduler_name    = self.name,
            selected_indices  = selected,
            selected_names    = names,
            utilities         = utilities,
            gains             = sel_gains,
            feasibilities     = sel_feas,
            costs             = sel_costs,
            weather           = constraint_engine.weather,
            time_used         = round(time_used, 2),
            time_budget       = 8.0,
            alpha_t           = round(alpha_t, 4),
            beta_t            = round(beta_t, 4),
            gamma             = round(gamma, 4),
            cum_sci_gain      = round(self.cumulative_gain, 4),
            mean_priority     = mean_priority,
            mean_sigma_before = sigma_before,
            mean_sigma_after  = sigma_after,
        )

    def get_logs_df(self) -> pd.DataFrame:
        if not self.logs:
            return pd.DataFrame()
        rows = []
        for log in self.logs:
            rows.append({
                "round":          log.round_number,
                "scheduler":      log.scheduler_name,
                "n_selected":     len(log.selected_indices),
                "weather":        log.weather,
                "time_used_hrs":  log.time_used,
                "alpha_t":        log.alpha_t,
                "beta_t":         log.beta_t,
                "gamma":          log.gamma,
                "cum_sci_gain":   log.cum_sci_gain,
                "mean_priority":  log.mean_priority,
                "mean_sigma_before": log.mean_sigma_before,
                "mean_sigma_after":  log.mean_sigma_after,
                "top_target":     log.selected_names[0] if log.selected_names else "",
            })
        return pd.DataFrame(rows)


# ── Baseline 1: Static Priority Scheduler ────────────────────────────────────
class StaticPriorityScheduler(BaseScheduler):
    """
    Baseline 1: Always selects planets with highest predicted priority score.
    No uncertainty or feasibility weighting.
    """

    def __init__(self, df: pd.DataFrame, initial_priorities: np.ndarray):
        super().__init__("Static Priority", df)
        self.static_scores = initial_priorities.copy()

    def select(self, simulator, constraint_engine, round_number, k, observed_set):
        cand_idx, feas, costs = self._feasible_candidates(constraint_engine, observed_set)
        if len(cand_idx) == 0:
            return [], []

        scores   = self.static_scores[cand_idx]
        order    = np.argsort(scores)[::-1]
        ranked   = cand_idx[order]
        r_costs  = costs[order]

        time_before = constraint_engine.time_budget
        selected, sel_costs = self._apply_budget(ranked, r_costs, constraint_engine, k)

        gains_arr = simulator.sigma[cand_idx] * simulator.detectability[cand_idx]
        log = self._build_log(round_number, selected, sel_costs, simulator,
                               constraint_engine, gains_arr, feas, cand_idx,
                               1.0, 1.0, 1.0, time_before)
        self.logs.append(log)
        return selected, sel_costs


# ── Baseline 2: Detectability Greedy Scheduler ───────────────────────────────
class DetectabilityGreedyScheduler(BaseScheduler):
    """
    Baseline 2: Always selects planets with highest detectability.
    """

    def select(self, simulator, constraint_engine, round_number, k, observed_set):
        cand_idx, feas, costs = self._feasible_candidates(constraint_engine, observed_set)
        if len(cand_idx) == 0:
            return [], []

        det    = simulator.detectability[cand_idx]
        order  = np.argsort(det)[::-1]
        ranked = cand_idx[order]
        r_costs = costs[order]

        time_before = constraint_engine.time_budget
        selected, sel_costs = self._apply_budget(ranked, r_costs, constraint_engine, k)

        gains_arr = simulator.sigma[cand_idx] * simulator.detectability[cand_idx]
        log = self._build_log(round_number, selected, sel_costs, simulator,
                               constraint_engine, gains_arr, feas, cand_idx,
                               0.0, 0.0, 1.0, time_before)
        self.logs.append(log)
        return selected, sel_costs


# ── Baseline 3: Uncertainty Greedy Scheduler ─────────────────────────────────
class UncertaintyGreedyScheduler(BaseScheduler):
    """
    Baseline 3: Always selects planets with highest prediction uncertainty.
    """

    def select(self, simulator, constraint_engine, round_number, k, observed_set):
        cand_idx, feas, costs = self._feasible_candidates(constraint_engine, observed_set)
        if len(cand_idx) == 0:
            return [], []

        uncert = simulator.sigma[cand_idx]
        order  = np.argsort(uncert)[::-1]
        ranked = cand_idx[order]
        r_costs = costs[order]

        time_before = constraint_engine.time_budget
        selected, sel_costs = self._apply_budget(ranked, r_costs, constraint_engine, k)

        gains_arr = simulator.sigma[cand_idx] * simulator.detectability[cand_idx]
        log = self._build_log(round_number, selected, sel_costs, simulator,
                               constraint_engine, gains_arr, feas, cand_idx,
                               1.0, 0.0, 0.0, time_before)
        self.logs.append(log)
        return selected, sel_costs


# ── Our Method: Adaptive Multi-Objective Scheduler ───────────────────────────
class AdaptiveScheduler(BaseScheduler):
    """
    Our method: Maximizes Utility = (Gain × Feasibility) / Cost

    Scientific Gain (Issue 3 — time-decaying exploration weight):
      Gain_i = alpha_t × U_i + beta_t × P_i + gamma × D_i

      alpha_t = alpha_0                    (uncertainty weight — constant)
      beta_t  = beta_0 × exp(-t / tau)    (priority weight — decays over time)
      gamma   = gamma_0                    (detectability weight — constant)

    This creates:
      - Early rounds: exploration (uncertainty-seeking, beta_t high)
      - Late rounds:  exploitation (priority-greedy, beta_t decayed)
    """

    def __init__(
        self,
        df:      pd.DataFrame,
        alpha_0: float = 0.50,   # uncertainty weight (constant)
        beta_0:  float = 0.30,   # priority weight at t=0
        gamma:   float = 0.20,   # detectability weight (constant)
        tau:     float = 15.0,   # exploration decay timescale (rounds)
    ):
        super().__init__("Adaptive Scheduler", df)
        self.alpha_0 = alpha_0
        self.beta_0  = beta_0
        self.gamma   = gamma
        self.tau     = tau

    def _get_weights(self, round_number: int) -> Tuple[float, float, float]:
        """
        Compute time-varying weights.
        beta_t decays exponentially; alpha and gamma remain constant.
        """
        alpha_t = self.alpha_0
        beta_t  = self.beta_0 * np.exp(-round_number / self.tau)
        gamma   = self.gamma
        # Re-normalize so weights sum to 1
        total   = alpha_t + beta_t + gamma
        return alpha_t / total, beta_t / total, gamma / total

    def _compute_gain(
        self,
        cand_idx:  np.ndarray,
        simulator: ObservationSimulator,
        alpha_t:   float,
        beta_t:    float,
        gamma:     float,
    ) -> np.ndarray:
        U = simulator.sigma[cand_idx]
        P = simulator.mu[cand_idx]
        D = simulator.detectability[cand_idx]
        return alpha_t * U + beta_t * P + gamma * D

    def select(self, simulator, constraint_engine, round_number, k, observed_set):
        cand_idx, feas, costs = self._feasible_candidates(constraint_engine, observed_set)
        if len(cand_idx) == 0:
            return [], []

        alpha_t, beta_t, gamma = self._get_weights(round_number)

        gains   = self._compute_gain(cand_idx, simulator, alpha_t, beta_t, gamma)
        utility = (gains * feas) / (costs + 1e-6)

        order   = np.argsort(utility)[::-1]
        ranked  = cand_idx[order]
        r_costs = costs[order]

        time_before = constraint_engine.time_budget
        selected, sel_costs = self._apply_budget(ranked, r_costs, constraint_engine, k)

        log = self._build_log(round_number, selected, sel_costs, simulator,
                               constraint_engine, gains, feas, cand_idx,
                               alpha_t, beta_t, gamma, time_before)
        self.logs.append(log)
        return selected, sel_costs


# ── Oracle Scheduler — perfect future knowledge upper bound ──────────────────
class OracleScheduler(AdaptiveScheduler):
    """
    Oracle Scheduler: selects the globally optimal set every round.

    Inherits from AdaptiveScheduler to share the dynamic weight scheduler,
    but overrides `_compute_gain` to use `true_priorities` (perfect future knowledge)
    instead of the noisy/estimated `mu` scores.

    Used as a true upper bound for regret and normalization.
    """

    def __init__(
        self,
        df:              pd.DataFrame,
        true_priorities: np.ndarray,
        alpha_0:         float = 0.50,
        beta_0:          float = 0.30,
        gamma:           float = 0.20,
        tau:             float = 15.0,
    ):
        super().__init__(df, alpha_0, beta_0, gamma, tau)
        self.name = "Oracle"
        self.true_priorities = true_priorities.copy()

    def _compute_gain(
        self,
        cand_idx:  np.ndarray,
        simulator: ObservationSimulator,
        alpha_t:   float,
        beta_t:    float,
        gamma:     float,
    ) -> np.ndarray:
        U = simulator.sigma[cand_idx]
        P = self.true_priorities[cand_idx]  # perfect ground truth knowledge!
        D = simulator.detectability[cand_idx]
        return alpha_t * U + beta_t * P + gamma * D



# ── Campaign Runner ───────────────────────────────────────────────────────────
def run_campaign(
    scheduler:          BaseScheduler,
    simulator:          ObservationSimulator,
    constraint_engine:  ObservationConstraintEngine,
    n_rounds:           int = 30,
    k_per_round:        int = 10,
    allow_reobs:        bool = False,
    verbose:            bool = True,
) -> dict:
    """
    Run a full observation campaign with a given scheduler.

    Parameters
    ----------
    scheduler          : any BaseScheduler subclass
    simulator          : ObservationSimulator (will be mutated)
    constraint_engine  : ObservationConstraintEngine (will be mutated)
    n_rounds           : int
    k_per_round        : int
    allow_reobs        : bool  -- allow re-observing the same planet
    verbose            : bool

    Returns
    -------
    dict with logs, observation history, final state
    """
    observed_set    = set()
    weather_history = []

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Campaign: {scheduler.name}")
        print(f"  {n_rounds} rounds x {k_per_round} planets/round")
        print(f"{'='*60}")

    for rnd in range(1, n_rounds + 1):
        constraint_engine.new_round()
        cs = constraint_engine.summary()

        selected, costs = scheduler.select(
            simulator         = simulator,
            constraint_engine = constraint_engine,
            round_number      = rnd,
            k                 = k_per_round,
            observed_set      = observed_set,
        )

        if selected:
            simulator.observe_batch(
                planet_indices = selected,
                round_number   = rnd,
                weather        = cs["weather"],
                costs          = np.array(costs),
            )
            observed_set.update(selected)

        weather_history.append(cs["weather"])

        if verbose and rnd % 5 == 0:
            log = scheduler.logs[-1] if scheduler.logs else None
            gain_str = f"{log.cum_sci_gain:.4f}" if log else "N/A"
            mu_str   = f"{log.mean_priority:.4f}" if log else "N/A"
            unc_sum  = simulator.uncertainty_summary()
            print(f"  Round {rnd:2d} | weather={cs['weather_desc']:<18} "
                  f"| cum_gain={gain_str} | mean_prio={mu_str} "
                  f"| observed={unc_sum['n_observed']}")

    if verbose:
        print(f"\n  Final: observed={len(observed_set)} planets | "
              f"cum_gain={scheduler.cumulative_gain:.4f} | "
              f"total_hrs={scheduler.total_time_used:.1f}")

    return {
        "scheduler_name":   scheduler.name,
        "logs_df":          scheduler.get_logs_df(),
        "obs_history_df":   simulator.get_history_df(),
        "final_state":      simulator.get_state(),
        "n_observed":       len(observed_set),
        "cumulative_gain":  scheduler.cumulative_gain,
        "total_time_used":  scheduler.total_time_used,
        "weather_history":  weather_history,
    }
