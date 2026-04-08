import numpy as np
from typing import List

def grade_calm_market(episode_rewards: List[float],
                      final_portfolio: float,
                      initial_portfolio: float = 10000.0) -> float:
    """Easy: reward positive return. Score in [0.0, 1.0]."""
    ret = (final_portfolio - initial_portfolio) / initial_portfolio
    # Map [-0.05, +0.05] return range to [0, 1]
    score = np.clip((ret + 0.05) / 0.10, 0.0, 1.0)
    return float(score)

def grade_volatile_market(episode_rewards: List[float],
                          survived: bool,
                          final_portfolio: float) -> float:
    """Medium: 50% survival, 50% Sharpe. Score in [0.0, 1.0]."""
    survival_score = 1.0 if survived else 0.0
    rewards = np.array(episode_rewards)
    sharpe = (rewards.mean() / (rewards.std() + 1e-8))
    sharpe_score = float(np.clip((sharpe + 1.0) / 2.0, 0.0, 1.0))
    return 0.5 * survival_score + 0.5 * sharpe_score

def grade_adversarial_market(episode_rewards: List[float],
                             survived: bool,
                             final_portfolio: float) -> float:
    """Hard: Sharpe under full adversary. Score in [0.0, 1.0]."""
    if not survived:
        return 0.0
    rewards = np.array(episode_rewards)
    sharpe = rewards.mean() / (rewards.std() + 1e-8)
    # Sharpe of 1.0 = perfect score; negative = 0
    score = float(np.clip(sharpe / 1.0, 0.0, 1.0))
    return score

GRADERS = {
    'calm-market': grade_calm_market,
    'volatile-market': grade_volatile_market,
    'adversarial-market': grade_adversarial_market,
}
