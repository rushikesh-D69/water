"""
constraint_engine.py — Stage 2
================================
Observation Constraint Engine

Models realistic telescope scheduling constraints per observation round:
  - Time budget (hrs/night, depleted per observation)
  - Observation cost (J-mag brightness, distance, integration time)
  - Visibility window (orbital-phase Gaussian availability)
  - Weather quality with AR(1) temporal autocorrelation (bad-weather streaks)
  - Telescope switching cost (slew time penalty)
  - Composite feasibility score F_i = visibility × weather × (1/cost) normalised

Design note (Issue 2):
  Weather uses an AR(1) process: w_t = rho*w_{t-1} + (1-rho)*Beta(5,2) + noise
  This creates realistic multi-round bad-weather streaks rather than iid noise.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ── Constants ────────────────────────────────────────────────────────────────
TELESCOPE_HRS_PER_NIGHT  = 8.0     # total available hours per round
BASE_OBS_COST_HRS        = 0.5     # minimum integration time [hrs]
MAX_OBS_COST_HRS         = 4.0     # maximum integration time [hrs]
SWITCH_COST_HRS          = 0.25    # telescope slew/reconfiguration penalty [hrs]

# AR(1) weather model parameters
WEATHER_RHO   = 0.65   # autocorrelation: 0=iid, 1=perfectly persistent
WEATHER_ALPHA = 5.0    # Beta shape alpha (skewed towards good weather)
WEATHER_BETA  = 2.0    # Beta shape beta


class WeatherModel:
    """
    AR(1) correlated weather quality model.

    w_t = rho * w_{t-1} + (1 - rho) * Beta(alpha, beta) + noise

    rho = 0.65 creates realistic multi-round bad-weather streaks.
    Output clipped to [0.05, 1.0] (never fully impossible, but degraded).
    """

    def __init__(self, rho=WEATHER_RHO, alpha=WEATHER_ALPHA, beta=WEATHER_BETA,
                 seed=42):
        self.rho   = rho
        self.alpha = alpha
        self.beta  = beta
        self.rng   = np.random.default_rng(seed)
        self.state = self.rng.beta(alpha, beta)   # initialise at steady-state draw

    def next_round(self):
        """Return weather quality for the next observation round."""
        iid_draw   = self.rng.beta(self.alpha, self.beta)
        noise      = self.rng.normal(0, 0.03)
        self.state = self.rho * self.state + (1 - self.rho) * iid_draw + noise
        self.state = float(np.clip(self.state, 0.05, 1.0))
        return self.state

    def describe(self):
        q = self.state
        if q >= 0.85: return f"Excellent ({q:.2f})"
        if q >= 0.65: return f"Good ({q:.2f})"
        if q >= 0.45: return f"Fair ({q:.2f})"
        return f"Poor ({q:.2f})"


class ObservationConstraintEngine:
    """
    Computes per-planet observation constraints and feasibility for each round.

    Attributes
    ----------
    time_budget   : float  -- remaining telescope hours in current round
    weather       : float  -- current round weather quality [0.05, 1.0]
    round_number  : int    -- current round index (1-based)
    """

    def __init__(self, df: pd.DataFrame, seed: int = 42):
        """
        Parameters
        ----------
        df   : pd.DataFrame
            Processed planet dataset (must contain 'sy_jmag', 'sy_dist',
            'pl_trandur', 'pl_orbper', 'detectability').
        seed : int
        """
        self.df           = df.reset_index(drop=True)
        self.n_planets    = len(df)
        self.rng          = np.random.default_rng(seed)
        self.weather_model = WeatherModel(seed=seed + 1)

        self.time_budget  = TELESCOPE_HRS_PER_NIGHT
        self.weather      = self.weather_model.state
        self.round_number = 0
        self.last_target  = None   # for switching cost

        self._compute_static_costs()
        self._compute_visibility_windows()

    # ─────────────────────────────────────────────────────────────────────────
    # Static (round-independent) computations
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_static_costs(self):
        """
        Observation cost [hrs] = integration time required for adequate SNR.
        Cost scales with stellar faintness and distance.

        cost_i = BASE + (J_mag - 8) * 0.15 + distance_factor
        Clipped to [BASE_OBS_COST_HRS, MAX_OBS_COST_HRS].
        """
        jmag  = self.df["sy_jmag"].fillna(self.df["sy_vmag"].fillna(11.0)).values
        dist  = self.df["sy_dist"].fillna(100.0).values
        tdur  = self.df["pl_trandur"].fillna(2.0).values.clip(0.5, 12.0)

        mag_cost   = np.clip((jmag - 8.0) * 0.15, 0, 2.0)
        dist_cost  = np.clip(np.log1p(dist / 50.0) * 0.3, 0, 1.5)
        tdur_cost  = tdur / 24.0   # transit duration as minimum integration

        raw_cost   = BASE_OBS_COST_HRS + mag_cost + dist_cost + tdur_cost
        self.obs_cost = np.clip(raw_cost, BASE_OBS_COST_HRS, MAX_OBS_COST_HRS)

    def _compute_visibility_windows(self):
        """
        Visibility probability modelled as a Gaussian over orbital phase.
        Planets with period < 10 days are observable most of the time.
        Longer-period planets have narrower observability windows.

        visibility_i: array shape (n_planets,) in [0.1, 1.0]
        This is the MEAN visibility — each round samples from this.
        """
        period = self.df["pl_orbper"].fillna(30.0).values.clip(0.1, None)
        # Short-period planets essentially always visible; longer period = narrower window
        # window_fraction in [0.05, 1.0]
        window_fraction = np.clip(10.0 / period, 0.05, 1.0)
        # Add a small floor so every planet has some chance
        self.visibility_mean = np.clip(window_fraction + 0.1, 0.1, 1.0)

    # ─────────────────────────────────────────────────────────────────────────
    # Round-level update
    # ─────────────────────────────────────────────────────────────────────────

    def new_round(self):
        """
        Advance to the next observation round.
        Resets time budget and samples new weather + visibility.
        """
        self.round_number += 1
        self.time_budget   = TELESCOPE_HRS_PER_NIGHT
        self.weather       = self.weather_model.next_round()
        self.last_target   = None

        # Sample per-round visibility: Bernoulli draw from mean visibility
        vis_draws = self.rng.uniform(0, 1, self.n_planets)
        self.visibility = np.where(
            vis_draws < self.visibility_mean, self.visibility_mean, 0.0
        )
        # Always keep at least 20% of planets visible
        n_visible = int(self.visibility.astype(bool).sum())
        if n_visible < max(20, self.n_planets // 10):
            # Force some visibility
            force_idx = np.argsort(self.visibility_mean)[::-1][:max(20, self.n_planets // 10)]
            self.visibility[force_idx] = self.visibility_mean[force_idx]

        return self

    # ─────────────────────────────────────────────────────────────────────────
    # Per-planet feasibility
    # ─────────────────────────────────────────────────────────────────────────

    def compute_feasibility(self, planet_indices=None):
        """
        Compute composite feasibility F_i for given planet indices.

        F_i = visibility_i × weather × (1 / normalised_cost_i)
        Normalised to [0, 1].

        Parameters
        ----------
        planet_indices : array-like or None
            Indices into df. If None, computes for all planets.

        Returns
        -------
        feasibility : np.ndarray shape (len(planet_indices),) in [0,1]
        cost        : np.ndarray shape (len(planet_indices),) in hours
        """
        if planet_indices is None:
            planet_indices = np.arange(self.n_planets)
        idx = np.array(planet_indices)

        vis    = self.visibility[idx]
        cost   = self.obs_cost[idx]
        w      = self.weather

        # Cost penalty: cheaper = more feasible
        cost_norm  = cost / MAX_OBS_COST_HRS          # [0,1], higher = more expensive
        cost_score = 1.0 - cost_norm                   # [0,1], higher = cheaper

        feasibility = vis * w * (0.5 + 0.5 * cost_score)
        # Normalize to [0, 1]
        max_f = feasibility.max()
        if max_f > 0:
            feasibility = feasibility / max_f
        feasibility = np.clip(feasibility, 0.0, 1.0)

        return feasibility, cost[np.arange(len(idx))]

    def can_observe(self, planet_idx, n_already_scheduled=0):
        """
        Check if observing planet_idx fits in remaining time budget.
        Includes switching cost if it's not the first target.
        """
        switch = SWITCH_COST_HRS if n_already_scheduled > 0 else 0.0
        total_cost = self.obs_cost[planet_idx] + switch
        return self.time_budget >= total_cost, total_cost

    def consume_budget(self, planet_idx, n_already_scheduled=0):
        """Deduct observation cost from time budget."""
        _, cost = self.can_observe(planet_idx, n_already_scheduled)
        self.time_budget -= cost
        self.time_budget  = max(0.0, self.time_budget)
        self.last_target  = planet_idx

    def summary(self):
        return {
            "round":        self.round_number,
            "weather":      round(self.weather, 3),
            "weather_desc": self.weather_model.describe(),
            "time_budget":  round(self.time_budget, 2),
            "n_visible":    int(self.visibility.astype(bool).sum()),
        }
