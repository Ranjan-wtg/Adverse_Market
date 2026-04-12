from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Union
import numpy as np
from openenv.core.env_server import Environment

from env.adverse_market_env import AdverseMarketEnv

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

class State(BaseModel):
    task_id: str
    step: int
    observation: Optional[Observation]
    regime: Optional[str]

class OpenEnvAdverseMarket(Environment):
    """OpenEnv-compliant wrapper around AdverseMarketEnv."""

    def __init__(self, task_id: str = 'adversarial-market',
                 adversary_policy=None):
        super().__init__()
        self._task_id = task_id
        self._env = AdverseMarketEnv(adversary_policy=adversary_policy)
        self._last_obs = None
        self._step_count = 0

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs) -> Observation:
        """Reset the environment."""
        obs_arr, _ = self._env.reset()
        self._step_count = 0
        self._last_obs = self._arr_to_obs(obs_arr)
        return self._last_obs

    def step(self, action: Union[Action, Dict[str, Any], int], timeout_s: Optional[float] = None, **kwargs):
        """Execute a step in the environment."""
        # Handle different action input types (OpenEnv can pass dicts or Pydantic models)
        if isinstance(action, Action):
            act_idx = action.action_index
        elif isinstance(action, dict):
            act_idx = action.get("action_index", 0)
        else:
            act_idx = int(action)

        ret = self._env.step(act_idx)
        if len(ret) == 5:
            obs_arr, r, terminated, truncated, info = ret
            done = terminated or truncated
        else:
            obs_arr, r, done, info = ret
            
        self._step_count += 1
        obs = self._arr_to_obs(obs_arr)
        
        # OpenEnv expects a float reward, we return it but store details in info
        reward_val = float(r)
        self._last_obs = obs
        
        # Optional: return a Reward object if the caller expects it, 
        # but the base class usually returns (obs, reward_val, done, info)
        return obs, reward_val, done, info

    @property
    def state(self) -> State:
        """Return the current environment state."""
        return State(
            task_id=self._task_id,
            step=self._step_count,
            observation=self._last_obs if self._last_obs else None,
            regime=self._env.price_proc.regime if hasattr(self._env, 'price_proc') else None
        )

    def close(self):
        if hasattr(self._env, 'close'):
            self._env.close()

    def _arr_to_obs(self, arr: np.ndarray) -> Observation:
        return Observation(
            price_returns=arr[:19].tolist(),
            position_norm=float(arr[19]),
            cash_norm=float(arr[20]),
            spread_ratio=float(arr[21]),
            volume_imbalance=float(arr[22]),
            portfolio_return=float(arr[23]),
            price_deviation=float(arr[24]),
            time_fraction=float(arr[25]))
