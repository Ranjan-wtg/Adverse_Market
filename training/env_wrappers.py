import gymnasium as gym

class BaseWrapper(gym.Wrapper):
    """Base wrapper for Stable Baselines 3 compatibility."""
    def __init__(self, env):
        super().__init__(env)
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info
        
class TraderEnv(BaseWrapper):
    """Exposes the standard Gym interface for training the Trader policy."""
    def __init__(self, env):
        super().__init__(env)
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def step(self, action):
        return self.env.step(action)

class AdversaryEnv(BaseWrapper):
    """
    Exposes the environment from the Adversary's perspective.
    Expects step(adversary_action) -> (adversary_obs, adversary_reward, done, info).
    """
    def __init__(self, env, trader_policy=None):
        super().__init__(env)
        from env.adverse_market_env import RandomAdversary
        # Actions 0-5
        self.action_space = gym.spaces.Discrete(6)
        # Obs: [regime_idx, drawdown, spread_mult] => 3 dims
        import numpy as np
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,))
        self.trader_policy = trader_policy

    def step(self, adversary_action):
        # The adversary's action is applied to the environment
        obs, trader_reward, done, trunc, info = self.env.step(self._get_trader_action())
        # We need to hack the underlying adversary to use standard RL interface,
        # but since we're wrapping the raw Env, we inject the perturbation BEFORE step_price
        
        # Reward is zero-sum
        reward = -trader_reward
        
        # New observation for adversary
        adv_obs = self.env._adversary_obs()
        return adv_obs, reward, done, trunc, info

    def _get_trader_action(self):
        if self.trader_policy is None:
            import random
            return random.randint(0, 8)
        else:
            # Assuming SB3 policy
            action, _ = self.trader_policy.predict(self.env._trader_obs(), deterministic=True)
            return action

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        return self.env._adversary_obs(), self.env._info()
