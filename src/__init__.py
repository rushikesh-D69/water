# Adaptive AI-Driven Telescope Target Prioritization Framework
# Stage 1: Static ML Habitability Predictor
# Stage 2: Adaptive Observation Scheduling Engine
# Stage 3: Reinforcement Learning Autonomous Scheduler
# Source package

# Stage 1 + 2 modules
from src.data_acquisition import *        # noqa: F401,F403
from src.ml_pipeline import *             # noqa: F401,F403
from src.constraint_engine import ObservationConstraintEngine, WeatherModel
from src.observation_simulator import ObservationSimulator
from src.scheduler import (
    BaseScheduler, StaticPriorityScheduler,
    DetectabilityGreedyScheduler, UncertaintyGreedyScheduler,
    AdaptiveScheduler, OracleScheduler, run_campaign,
)
from src.evaluation import *              # noqa: F401,F403

# Stage 3 RL modules (requires: gymnasium, stable-baselines3, sb3-contrib)
from src.rl_environment import ExoplanetSchedulingEnv, make_shortlist
from src.rl_agent import (
    collect_bc_trajectories, pretrain_bc,
    train_ppo_curriculum, evaluate_policy_campaigns,
    evaluate_generalization, explain_policy_decision,
    LinUCBBandit, RLScheduler,
)
from src.rl_evaluation import (
    plot_training_reward_curve, plot_policy_heatmap,
    plot_reward_decomposition, plot_exploration_exploitation_timeline,
    plot_extended_pareto, plot_generalization, plot_state_tsne,
    print_policy_explanation, build_stage3_comparison_table,
    save_stage3_comparison,
)
