import numpy as np
import pandas as pd
from typing import List, Tuple
from stable_baselines3.common.base_class import BaseAlgorithm

from src.scheduler import BaseScheduler, RoundLog
from src.observation_simulator import ObservationSimulator
from src.constraint_engine import ObservationConstraintEngine

class RLScheduler(BaseScheduler):
    """
    Scheduler that uses a trained Reinforcement Learning model (e.g., MaskablePPO) 
    to select the best sequence of planets to observe per round.
    """
    def __init__(self, name: str, df: pd.DataFrame, model: BaseAlgorithm, top_k: int = 100):
        super().__init__(name, df)
        self.model = model
        self.top_k = min(top_k, len(df))
        
        # Find the top_k indices based on initial priority (same as environment logic)
        self.target_indices = np.argsort(np.asarray(df["priority_score"]))[::-1][:self.top_k]

    def _get_obs(self, round_number: int, simulator: ObservationSimulator, 
                 constraint_engine: ObservationConstraintEngine, 
                 observed_set: set, local_observed: set) -> np.ndarray:
        """Construct the Gym environment state array for the model to predict on."""
        obs_dim = 2 + (4 * self.top_k)
        obs = np.zeros(obs_dim, dtype=np.float32)
        
        max_rounds = 30
        obs[0] = (max_rounds - round_number) / max_rounds
        obs[1] = constraint_engine.weather
        
        offset = 2
        for i, idx in enumerate(self.target_indices):
            obs[offset + i*4 + 0] = simulator.mu[idx]
            obs[offset + i*4 + 1] = simulator.sigma[idx]
            obs[offset + i*4 + 2] = simulator.detectability[idx]
            
            vis = 1.0 if constraint_engine.visibility[idx] > 0 and idx not in observed_set and idx not in local_observed else 0.0
            obs[offset + i*4 + 3] = vis
            
        return obs

    def _get_action_masks(self, constraint_engine: ObservationConstraintEngine, 
                          observed_set: set, local_observed: set, n_selected: int) -> np.ndarray:
        masks = np.zeros(self.top_k + 1, dtype=bool)
        masks[self.top_k] = True # Skip action is always valid
        
        for i, idx in enumerate(self.target_indices):
            if idx not in observed_set and idx not in local_observed and constraint_engine.visibility[idx] > 0:
                can_observe, _ = constraint_engine.can_observe(idx, n_selected)
                if can_observe:
                    masks[i] = True
                    
        return masks

    def select(self, simulator: ObservationSimulator, constraint_engine: ObservationConstraintEngine, 
               round_number: int, k: int, observed_set: set) -> Tuple[List[int], List[float]]:
        
        cand_idx, feas, costs = self._feasible_candidates(constraint_engine, observed_set)
        if len(cand_idx) == 0:
            return [], []

        selected = []
        sel_costs = []
        local_observed = set()
        
        time_before = constraint_engine.time_budget
        temp_budget = time_before
        n_selected = 0
        
        # Sequentially query the RL agent for actions until budget or K is reached
        for _ in range(self.top_k):
            if n_selected >= k or temp_budget <= 0:
                break
                
            obs = self._get_obs(round_number, simulator, constraint_engine, observed_set, local_observed)
            action_masks = self._get_action_masks(constraint_engine, observed_set, local_observed, n_selected)
            
            # Predict action deterministically using action_masks
            action, _ = self.model.predict(obs, action_masks=action_masks, deterministic=True)
            
            # Action == top_k means "skip"
            if action == self.top_k:
                break
                
            planet_idx = self.target_indices[action]
            
            # Stop if model predicts an invalid action (avoids infinite loop)
            # This shouldn't happen with valid masking, but kept as a safety fallback
            if planet_idx in observed_set or planet_idx in local_observed or constraint_engine.visibility[planet_idx] <= 0:
                break
                
            can_obs, cost = constraint_engine.can_observe(planet_idx, n_selected)
            if not can_obs or temp_budget - cost < 0:
                break
                
            selected.append(int(planet_idx))
            sel_costs.append(cost)
            local_observed.add(planet_idx)
            temp_budget -= cost
            n_selected += 1
            
        # Consume the budget in the actual constraint engine
        for i, idx in enumerate(selected):
            constraint_engine.consume_budget(idx, i)
            
        # Build logs
        gains_arr = simulator.sigma[cand_idx] * simulator.detectability[cand_idx]
        log = self._build_log(
            round_number=round_number,
            selected=selected,
            sel_costs=sel_costs,
            simulator=simulator,
            constraint_engine=constraint_engine,
            gains_arr=gains_arr,
            feas_arr=feas,
            cand_idx=cand_idx,
            alpha_t=0.0,  # RL doesn't use explicit weights
            beta_t=0.0,
            gamma=0.0,
            time_before=time_before
        )
        self.logs.append(log)
        
        return selected, sel_costs
