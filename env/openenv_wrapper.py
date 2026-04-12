"""OpenEnv-compliant wrapper around AdverseMarketEnv.

Uses the official openenv-core base types so that create_app() can
auto-register /health, /metadata, /schema, /reset, /step, /state.
"""

from typing import Any, Dict, List, Optional
import numpy as np

from openenv.core.env_server import Environment
from openenv.core.env_server.types import (
    Action as OpenEnvAction,
    Observation as OpenEnvObservation,
    State as OpenEnvState,
)

from env.adverse_market_env import AdverseMarketEnv


# ---------------------------------------------------------------------------
# Pydantic models that inherit from the official OpenEnv base types
# ---------------------------------------------------------------------------

class AdverseMarketAction(OpenEnvAction):
    """Action for the AdverseMarket environment."""
    action_index: int = 0  # 0-8 discrete action


class AdverseMarketObservation(OpenEnvObservation):
    """Observation returned by the AdverseMarket environment."""
    price_returns: List[float] = []
    position_norm: float = 0.0
    cash_norm: float = 1.0
    spread_ratio: float = 1.0
    volume_imbalance: float = 0.0
    portfolio_return: float = 0.0
    price_deviation: float = 0.0
    time_fraction: float = 0.0


class AdverseMarketState(OpenEnvState):
    """State of the AdverseMarket environment."""
    task_id: str = "adversarial-market"
    regime: Optional[str] = None


# ---------------------------------------------------------------------------
# Environment class (inherits from openenv.core.env_server.Environment)
# ---------------------------------------------------------------------------

class AdverseMarketEnvironment(Environment):
    """OpenEnv-compliant AdverseMarket environment."""

    def __init__(self, task_id: str = "adversarial-market",
                 adversary_policy=None, **kwargs):
        super().__init__(**kwargs)
        self._task_id = task_id
        self._env = AdverseMarketEnv(adversary_policy=adversary_policy)
        self._last_obs: Optional[AdverseMarketObservation] = None
        self._step_count = 0

    # -- required by Environment ABC -----------------------------------------

    def reset(self, seed: Optional[int] = None,
              episode_id: Optional[str] = None,
              **kwargs: Any) -> AdverseMarketObservation:
        obs_arr, _ = self._env.reset()
        self._step_count = 0
        self._last_obs = self._arr_to_obs(obs_arr, reward=None, done=False)
        return self._last_obs

    def step(self, action: AdverseMarketAction,
             timeout_s: Optional[float] = None,
             **kwargs: Any) -> AdverseMarketObservation:
        act_idx = action.action_index

        ret = self._env.step(act_idx)
        if len(ret) == 5:
            obs_arr, r, terminated, truncated, info = ret
            done = terminated or truncated
        else:
            obs_arr, r, done, info = ret

        self._step_count += 1
        self._last_obs = self._arr_to_obs(obs_arr,
                                          reward=float(r),
                                          done=done,
                                          info=info)
        return self._last_obs

    @property
    def state(self) -> AdverseMarketState:
        return AdverseMarketState(
            task_id=self._task_id,
            step_count=self._step_count,
            regime=(self._env.price_proc.regime
                    if hasattr(self._env, "price_proc") else None),
        )

    def close(self):
        if hasattr(self._env, "close"):
            self._env.close()

    # -- helpers --------------------------------------------------------------

    @staticmethod
    def _arr_to_obs(arr: np.ndarray,
                    reward=None,
                    done: bool = False,
                    info: Optional[Dict] = None) -> AdverseMarketObservation:
        return AdverseMarketObservation(
            price_returns=arr[:19].tolist(),
            position_norm=float(arr[19]),
            cash_norm=float(arr[20]),
            spread_ratio=float(arr[21]),
            volume_imbalance=float(arr[22]),
            portfolio_return=float(arr[23]),
            price_deviation=float(arr[24]),
            time_fraction=float(arr[25]),
            reward=reward,
            done=done,
            metadata=info or {},
        )
