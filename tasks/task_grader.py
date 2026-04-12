import numpy as np
from typing import List

def strictly_between_0_and_1(value: float) -> float:
    """Squashes a raw score [0, 1] into a strict range [0.01, 0.99] as per validator rules."""
    clipped = np.clip(value, 0.0, 1.0)
    return float(0.01 + 0.98 * clipped)

def grade_calm_market(episode_rewards: List[float],
                      final_portfolio: float,
                      initial_portfolio: float = 10000.0) -> float:
    """Easy: reward positive return. Score squashed to (0, 1)."""
    ret = (final_portfolio - initial_portfolio) / initial_portfolio
    # Map [-0.05, +0.05] return range to [0, 1]
    raw_score = (ret + 0.05) / 0.10
    return strictly_between_0_and_1(raw_score)

def grade_volatile_market(episode_rewards: List[float],
                          survived: bool,
                          final_portfolio: float) -> float:
    """Medium: 50% survival, 50% Sharpe. Score squashed to (0, 1)."""
    survival_score = 1.0 if survived else 0.0
    rewards = np.array(episode_rewards)
    if len(rewards) > 0 and rewards.std() > 0:
        sharpe = rewards.mean() / (rewards.std() + 1e-8)
        sharpe_score = (sharpe + 1.0) / 2.0
    else:
        sharpe_score = 0.0
    
    raw_score = 0.5 * survival_score + 0.5 * sharpe_score
    return strictly_between_0_and_1(raw_score)

def grade_adversarial_market(episode_rewards: List[float],
                              survived: bool,
                              final_portfolio: float) -> float:
    """Hard: Sharpe under full adversary. Score squashed to (0, 1)."""
    if not survived:
        return strictly_between_0_and_1(0.0)
    
    rewards = np.array(episode_rewards)
    if len(rewards) > 0 and rewards.std() > 0:
        sharpe = rewards.mean() / (rewards.std() + 1e-8)
        sharpe_score = sharpe / 1.0  # Sharpe of 1.0 = top score
    else:
        sharpe_score = 0.0
        
    return strictly_between_0_and_1(sharpe_score)

GRADERS = {
    'calm-market': grade_calm_market,
    'volatile-market': grade_volatile_market,
    'adversarial-market': grade_adversarial_market,
}
