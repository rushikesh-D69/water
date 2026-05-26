"""
observation_simulator.py — Stage 2
====================================
Observation Simulator

Simulates what a telescope observation reveals about a planet.

Design notes (Issue 4):
  Noise is NOT uniform — it depends on:
    - Detectability D_i     : high D = cleaner measurement
    - Weather quality w_t   : bad weather degrades SNR
    - SNR proxy             : directly scales noise level
    - Number of observations: diminishing returns (sqrt law)

  Uncertainty update model (Bayesian-inspired):
    sigma_new = sigma_old * noise_factor / sqrt(n_obs)

  Priority update:
    mu_new = mu_old + N(0, sigma_measurement)
    where sigma_measurement = base_noise * (1/SNR) * (1/weather)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ── Observation record ────────────────────────────────────────────────────────
@dataclass
class ObservationRecord:
    planet_idx:    int
    planet_name:   str
    round_number:  int
    mu_before:     float
    sigma_before:  float
    mu_after:      float
    sigma_after:   float
    weather:       float
    detectability: float
    cost_hrs:      float
    snr_effective: float
    measurement_delta: float
    obs_number:    int    # how many times this planet has been observed


# ── Simulator ─────────────────────────────────────────────────────────────────
class ObservationSimulator:
    """
    Simulates the outcome of telescope observations on planet state
    (predicted priority mean and uncertainty).

    Parameters
    ----------
    df             : pd.DataFrame  -- full processed planet dataframe
    initial_means  : np.ndarray    -- predicted priority scores from Stage 1
    initial_sigmas : np.ndarray    -- prediction uncertainties from Stage 1
    seed           : int
    """

    def __init__(
        self,
        df:             pd.DataFrame,
        initial_means:  np.ndarray,
        initial_sigmas: np.ndarray,
        seed:           int = 42,
    ):
        self.df             = df.reset_index(drop=True)
        self.n_planets      = len(df)
        self.rng            = np.random.default_rng(seed)

        # Mutable state — always numpy float64 regardless of input type
        self.mu    = np.asarray(initial_means,  dtype=np.float64).ravel().copy()
        self.sigma = np.asarray(initial_sigmas, dtype=np.float64).ravel().copy()

        # Track how many times each planet has been observed
        self.obs_count: np.ndarray = np.zeros(self.n_planets, dtype=int)

        # Full history log
        self.history: List[ObservationRecord] = []

        # Pre-extract stable physical quantities
        self.detectability = np.asarray(df["detectability"].fillna(0.1).values, dtype=np.float64)
        self.snr_proxy     = np.asarray(df["snr_proxy"].fillna(0.01).values,    dtype=np.float64)
        snr_99             = np.percentile(self.snr_proxy[self.snr_proxy > 0], 99)
        self.snr_norm      = np.clip(self.snr_proxy / (snr_99 + 1e-8), 0.01, 1.0)

    # ── Core observation method ───────────────────────────────────────────────

    def observe(
        self,
        planet_idx:   int,
        round_number: int,
        weather:      float,
        cost_hrs:     float,
    ) -> ObservationRecord:
        """
        Simulate one observation of planet `planet_idx`.

        Uncertainty update (Issue 4 — noise depends on detectability, SNR, weather):
          sigma_new = sigma_old × reduction_factor / sqrt(n_obs)

          reduction_factor = 0.4 + 0.4 × detectability + 0.2 × weather
            → best case (D=1, w=1): factor = 1.0 → halved every obs (with sqrt law)
            → worst case (D=0, w=0): factor ≈ 0.4 → smaller reduction

        Priority update:
          delta ~ N(0, sigma_measurement)
          sigma_measurement = 0.05 × (1 / SNR_norm) × (1 / weather)

        Returns
        -------
        ObservationRecord with full before/after state.
        """
        idx = int(planet_idx)
        n   = self.obs_count[idx] + 1   # new observation count

        mu_old    = float(self.mu[idx])
        sigma_old = float(self.sigma[idx])
        D         = float(self.detectability[idx])
        snr       = float(self.snr_norm[idx])
        w         = float(np.clip(weather, 0.05, 1.0))

        # ── Uncertainty reduction ─────────────────────────────────────────────
        # Better detectability + better weather → larger uncertainty reduction
        reduction = 0.4 + 0.4 * D + 0.2 * w       # in [0.4, 1.0]
        sigma_new = sigma_old * reduction / np.sqrt(n)
        sigma_new = float(np.clip(sigma_new, 0.01, sigma_old))

        # ── Effective SNR (degraded by weather) ──────────────────────────────
        snr_effective = snr * w * D

        # ── Priority measurement noise ────────────────────────────────────────
        # Low SNR + bad weather → noisier measurement → larger delta
        sigma_meas = 0.05 / (snr_effective + 0.05)   # measurement noise std
        delta      = float(self.rng.normal(0, sigma_meas))

        mu_new = float(np.clip(mu_old + delta, 0.0, 1.0))

        # ── Update state ──────────────────────────────────────────────────────
        self.mu[idx]        = mu_new
        self.sigma[idx]     = sigma_new
        self.obs_count[idx] = n

        # ── Record ────────────────────────────────────────────────────────────
        planet_name = str(self.df["pl_name"].iloc[idx]) if "pl_name" in self.df.columns else f"Planet_{idx}"

        rec = ObservationRecord(
            planet_idx        = idx,
            planet_name       = planet_name,
            round_number      = round_number,
            mu_before         = mu_old,
            sigma_before      = sigma_old,
            mu_after          = mu_new,
            sigma_after       = sigma_new,
            weather           = w,
            detectability     = D,
            cost_hrs          = cost_hrs,
            snr_effective     = snr_effective,
            measurement_delta = delta,
            obs_number        = n,
        )
        self.history.append(rec)
        return rec

    # ── Bulk observation ──────────────────────────────────────────────────────

    def observe_batch(
        self,
        planet_indices: List[int],
        round_number:   int,
        weather:        float,
        costs:          np.ndarray,
    ) -> List[ObservationRecord]:
        """Observe a batch of planets in one round."""
        records = []
        for i, idx in enumerate(planet_indices):
            rec = self.observe(
                planet_idx   = idx,
                round_number = round_number,
                weather      = weather,
                cost_hrs     = float(costs[i]) if i < len(costs) else 1.0,
            )
            records.append(rec)
        return records

    # ── State access ──────────────────────────────────────────────────────────

    def get_state(self) -> pd.DataFrame:
        """Return current state of all planets as a DataFrame."""
        return pd.DataFrame({
            "pl_name":      self.df["pl_name"].values if "pl_name" in self.df else np.arange(self.n_planets),
            "mu":           self.mu,
            "sigma":        self.sigma,
            "obs_count":    self.obs_count,
            "detectability": self.detectability,
            "scientific_gain": self.sigma * self.detectability,
        })

    def get_history_df(self) -> pd.DataFrame:
        """Return full observation history as a DataFrame."""
        if not self.history:
            return pd.DataFrame()
        rows = []
        for r in self.history:
            rows.append({
                "round":          r.round_number,
                "planet_idx":     r.planet_idx,
                "planet_name":    r.planet_name,
                "mu_before":      r.mu_before,
                "sigma_before":   r.sigma_before,
                "mu_after":       r.mu_after,
                "sigma_after":    r.sigma_after,
                "delta":          r.measurement_delta,
                "sigma_reduction": r.sigma_before - r.sigma_after,
                "weather":        r.weather,
                "detectability":  r.detectability,
                "snr_effective":  r.snr_effective,
                "cost_hrs":       r.cost_hrs,
                "obs_number":     r.obs_number,
            })
        return pd.DataFrame(rows)

    def uncertainty_summary(self) -> dict:
        observed_mask   = self.obs_count > 0
        unobserved_mask = ~observed_mask
        return {
            "n_observed":          int(observed_mask.sum()),
            "n_unobserved":        int(unobserved_mask.sum()),
            "mean_sigma_observed": float(self.sigma[observed_mask].mean()) if observed_mask.any() else 0.0,
            "mean_sigma_unobs":    float(self.sigma[unobserved_mask].mean()) if unobserved_mask.any() else 0.0,
            "total_sigma_reduced": float((self.sigma[observed_mask] < 0.1).sum()),
        }
