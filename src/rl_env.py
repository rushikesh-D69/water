import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Optional, Tuple
from src.observation_simulator import ObservationSimulator
from src.constraint_engine import ObservationConstraintEngine

class ExoplanetSchedulingEnv(gym.Env):
    """
    OpenAI Gymnasium Environment for Exoplanet Observation Scheduling.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        simulator: ObservationSimulator,
        constraint_engine: ObservationConstraintEngine,
        max_rounds: int = 30,
        top_k_targets: int = 100,
    ):
        super().__init__()
        self.df = df
        self.simulator = simulator
        self.constraint_engine = constraint_engine
        self.max_rounds = max_rounds
        self.top_k = min(top_k_targets, len(df))
        
        self.initial_mu = self.simulator.mu.copy()
        self.initial_sigma = self.simulator.sigma.copy()
        
        # We only consider the top_k targets based on initial priority to keep action space reasonable
        # We assume initial_means are sorted or we just take the first top_k from the simulator
        self.target_indices = np.argsort(self.simulator.mu)[::-1][:self.top_k]
        
        # Action Space: which of the top_k targets to observe next
        # Action top_k means "skip to next round" (if agent decides to save time or no good targets)
        self.action_space = spaces.Discrete(self.top_k + 1)
        
        # State Space: 
        # [round_remaining, weather] + [mu, sigma, detectability, visibility] * top_k
        self.obs_dim = 2 + (4 * self.top_k)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )
        
        self.current_round = 1
        self.n_selected_this_round = 0
        self.cumulative_gain = 0.0
        self.observed_set = set()

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        
        # Reset internal state
        self.current_round = 1
        self.n_selected_this_round = 0
        self.cumulative_gain = 0.0
        self.observed_set = set()
        
        # Reset constraint engine and simulator to initial states
        if seed is not None:
            self.constraint_engine.weather_model = __import__('src.constraint_engine', fromlist=['WeatherModel']).WeatherModel(seed=seed)
        else:
            self.constraint_engine.weather_model = __import__('src.constraint_engine', fromlist=['WeatherModel']).WeatherModel()
        self.constraint_engine.new_round()
        
        # Re-initialize simulator state (mu and sigma)
        self.simulator.mu = self.initial_mu.copy()
        self.simulator.sigma = self.initial_sigma.copy()
        self.simulator.obs_count = np.zeros(self.simulator.n_planets, dtype=int)
        self.simulator.history = []

        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        obs[0] = (self.max_rounds - self.current_round) / self.max_rounds
        obs[1] = self.constraint_engine.weather
        
        offset = 2
        for i, idx in enumerate(self.target_indices):
            obs[offset + i*4 + 0] = self.simulator.mu[idx]
            obs[offset + i*4 + 1] = self.simulator.sigma[idx]
            obs[offset + i*4 + 2] = self.simulator.detectability[idx]
            
            # Visibility: 1 if visible and not already observed, else 0
            vis = 1.0 if self.constraint_engine.visibility[idx] > 0 and idx not in self.observed_set else 0.0
            obs[offset + i*4 + 3] = vis
            
        return obs

    def action_masks(self) -> np.ndarray:
        masks = np.zeros(self.top_k + 1, dtype=bool)
        # Skip action is always valid
        masks[self.top_k] = True
        
        for i, idx in enumerate(self.target_indices):
            if idx not in self.observed_set and self.constraint_engine.visibility[idx] > 0:
                can_observe, _ = self.constraint_engine.can_observe(idx, self.n_selected_this_round)
                if can_observe:
                    masks[i] = True
                    
        return masks

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        reward = 0.0
        terminated = False
        truncated = False
        
        # Skip action
        if action == self.top_k:
            self._advance_round()
            if self.current_round > self.max_rounds:
                terminated = True
            return self._get_obs(), reward, terminated, truncated, {}
        
        planet_idx = self.target_indices[action]
        
        # Check constraints
        if planet_idx in self.observed_set or self.constraint_engine.visibility[planet_idx] <= 0:
            reward = -1.0 # Penalty for invalid action
            # Force advance round to prevent infinite loop of bad actions
            self._advance_round()
        else:
            can_observe, cost = self.constraint_engine.can_observe(planet_idx, self.n_selected_this_round)
            if not can_observe:
                self._advance_round()
            else:
                self.constraint_engine.consume_budget(planet_idx, self.n_selected_this_round)
                
                # Observe
                weather = self.constraint_engine.weather
                rec = self.simulator.observe(planet_idx, self.current_round, weather, cost)
                self.observed_set.add(planet_idx)
                self.n_selected_this_round += 1
                
                # Reward: We want to maximize uncertainty reduction + priority
                # This aligns with the AdaptiveScheduler gain function
                # sigma * detectability is the "gain" in the baseline
                gain = rec.sigma_before * rec.detectability
                self.cumulative_gain += gain
                reward = gain
                
                if self.constraint_engine.time_budget <= 0:
                    self._advance_round()

        if self.current_round > self.max_rounds:
            terminated = True
            # Terminal bonus
            reward += self.cumulative_gain * 0.1

        return self._get_obs(), reward, terminated, truncated, {}

    def _advance_round(self):
        self.current_round += 1
        self.n_selected_this_round = 0
        if self.current_round <= self.max_rounds:
            self.constraint_engine.new_round()
