from pydantic import BaseModel
from typing import Optional, List
import numpy as np

# We'll import to a mock if the true env isn't provided yet
try:
    from env.adverse_market_env import AdverseMarketEnv
except ImportError:
    # MOCK ENVIRONMENT since base code isn't provided.
    class AdverseMarketEnv:
        def __init__(self, adversary_policy=None):
            self.cash = 10000.0
            self.position = 0.0
            self.price_hist = [100.0]
            class MockPriceProc:
                regime = "bull"
            self.price_proc = MockPriceProc()
        def reset(self):
            return np.zeros(26), {}
        def step(self, action):
            return np.zeros(26), 0.1, False, False, {'pnl': 0.1}
        def close(self):
            pass

class Observation(BaseModel):
    price_returns: List[float]       # 19 rolling log-returns
    position_norm: float             # position / 100
    cash_norm: float                 # cash / 10000
    spread_ratio: float              # spread / base_spread
    volume_imbalance: float
    portfolio_return: float
    price_deviation: float
    time_fraction: float             # t / max_steps

class Action(BaseModel):
    action_index: int                # 0-8 discrete action

class Reward(BaseModel):
    value: float
    pnl_component: float
    inventory_penalty: float
    drawdown_penalty: float
    transaction_cost: float

class OpenEnvAdverseMarket:
    """OpenEnv-compliant wrapper around AdverseMarketEnv."""

    def __init__(self, task_id: str = 'adversarial-market',
                 adversary_policy=None):
        self._task_id = task_id
        self._env = AdverseMarketEnv(adversary_policy=adversary_policy)
        self._last_obs = None
        self._step_count = 0

    def reset(self) -> Observation:
        obs_arr, _ = self._env.reset()
        self._step_count = 0
        self._last_obs = self._arr_to_obs(obs_arr)
        return self._last_obs

    def step(self, action: Action):
        # Gym might return 5 values (obs, reward, terminated, truncated, info)
        ret = self._env.step(action.action_index)
        if len(ret) == 5:
            obs_arr, r, terminated, truncated, info = ret
            done = terminated or truncated
        else:
            obs_arr, r, done, info = ret
        self._step_count += 1
        obs = self._arr_to_obs(obs_arr)
        reward = Reward(
            value=float(r), pnl_component=float(info.get('pnl', 0.0)),
            inventory_penalty=0.0, drawdown_penalty=0.0,
            transaction_cost=0.0)
        self._last_obs = obs
        return obs, reward, done, info

    def state(self) -> dict:
        return {
            'task_id': self._task_id,
            'step': self._step_count,
            'observation': self._last_obs.model_dump()
                          if self._last_obs else None,
            'regime': self._env.price_proc.regime
                      if hasattr(self._env, 'price_proc') else None,
        }

    def close(self):
        if hasattr(self._env, 'close'):
            self._env.close()

    def _arr_to_obs(self, arr: np.ndarray) -> Observation:
        # Handling potentially smaller arrays from mock safely
        pad = np.zeros(26)
        if len(arr) < 26:
            pad[:len(arr)] = arr
            arr = pad
        return Observation(
            price_returns=arr[:19].tolist(),
            position_norm=float(arr[19]),
            cash_norm=float(arr[20]),
            spread_ratio=float(arr[21]),
            volume_imbalance=float(arr[22]),
            portfolio_return=float(arr[23]),
            price_deviation=float(arr[24]),
            time_fraction=float(arr[25]))
