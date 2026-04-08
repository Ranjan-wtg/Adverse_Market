import gymnasium as gym
import numpy as np

class RandomAdversary:
    def act(self, obs):
        return np.random.choice(6)

class AdverseMarketEnv(gym.Env):
    """
    AdverseMarket-v0: Gym environment with learned adversarial market.
    """
    REGIMES = ['bull', 'bear', 'crisis', 'chop']
    REGIME_DRIFT = {'bull': 0.01, 'bear': -0.01, 'crisis': -0.05, 'chop': 0.0}

    def __init__(self, adversary_policy=None, lam=0.05, kap=0.15, tau=0.001):
        self.adversary = adversary_policy or RandomAdversary()
        self.lam = lam
        self.kap = kap
        self.tau = tau
        self.action_space = gym.spaces.Discrete(9)   # 3 actions × 3 sizes
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(26,))
            
        self.max_steps = 1000
        # Internal state tracking
        self.cash = 10000.0
        self.position = 0.0
        self.price_hist = [100.0] * 20
        self.t = 0
        self.base_sigma = 0.02
        self.sigma_mult = 1.0
        self.base_spread = 0.05
        self.spread_mult = 1.0
        
        self.regime_idx = 0
        self.drawdown = 0.0
        self.peak_value = 10000.0
        self.transaction_cost = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.cash = 10000.0
        self.position = 0.0
        self.price_hist = [100.0] * 20
        self.t = 0
        self.sigma_mult = 1.0
        self.spread_mult = 1.0
        self.regime_idx = 0
        self.drawdown = 0.0
        self.peak_value = 10000.0
        self.transaction_cost = 0.0
        return self._trader_obs(), self._info()

    def step(self, trader_action):
        self.t += 1
        
        # 1. Adversary selects perturbation
        adv_obs = self._adversary_obs()
        adv_action = self.adversary.act(adv_obs) if hasattr(self.adversary, 'act') else 0
        self._apply_adversary(adv_action)

        # 2. Evolve price process
        self._step_price()

        # 3. Execute trader action
        delta_pnl = self._execute_trade(trader_action)

        # 4. Compute reward
        reward = (delta_pnl
                  - self.lam * abs(self.position)
                  - self.kap * max(0, self.drawdown)
                  - self.tau * self.transaction_cost)

        obs = self._trader_obs()
        done = self.cash <= 0 or self.t >= self.max_steps
        return obs, reward, done, False, self._info()

    def _apply_adversary(self, action):
        if action == 0:   self.sigma_mult = 1.0   
        elif action == 1: self.sigma_mult = 2.0   
        elif action == 2: self.sigma_mult = 5.0   
        elif action == 3: self._shift_regime()    
        elif action == 4: self.spread_mult = 2.0  
        elif action == 5: self._flash_crash(0.03) 

    def _shift_regime(self):
        self.regime_idx = (self.regime_idx + 1) % len(self.REGIMES)

    def _flash_crash(self, drop):
        current_price = self.price_hist[-1]
        self.price_hist.append(current_price * (1.0 - drop))
        self.price_hist.pop(0)

    def _step_price(self):
        current_price = self.price_hist[-1]
        regime_name = self.REGIMES[self.regime_idx]
        mu = self.REGIME_DRIFT[regime_name]
        sigma = self.base_sigma * self.sigma_mult
        
        # GBM step
        dt = 1.0 / 252.0
        dW = np.random.normal(0, np.sqrt(dt))
        np_price = current_price * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW)
        self.price_hist.append(float(np_price))
        self.price_hist.pop(0)

    def _execute_trade(self, action):
        current_price = self.price_hist[-1]
        prev_value = self.cash + self.position * current_price
        spread = self.base_spread * self.spread_mult
        buy_price = current_price + spread / 2
        sell_price = current_price - spread / 2
        
        trade_qty = 0.0
        if action == 1: trade_qty = (self.cash * 0.1) / buy_price
        elif action == 2: trade_qty = (self.cash * 0.2) / buy_price
        elif action == 3: trade_qty = (self.cash * 0.33) / buy_price
        elif action == 4: trade_qty = -(self.position * 0.1)
        elif action == 5: trade_qty = -(self.position * 0.2)
        elif action == 6: trade_qty = -(self.position * 0.33)
        elif action == 7: trade_qty = self.cash / buy_price
        elif action == 8: trade_qty = -self.position

        if trade_qty > 0:
            cost = trade_qty * buy_price
            if cost <= self.cash:
                self.cash -= cost
                self.position += trade_qty
                self.transaction_cost = cost * self.tau
        elif trade_qty < 0:
            revenue = abs(trade_qty) * sell_price
            if abs(trade_qty) <= self.position:
                self.cash += revenue
                self.position += trade_qty
                self.transaction_cost = revenue * self.tau
        else:
            self.transaction_cost = 0.0

        current_value = self.cash + self.position * current_price
        if current_value > self.peak_value:
            self.peak_value = current_value
        
        self.drawdown = (self.peak_value - current_value) / self.peak_value
        return current_value - prev_value

    def _trader_obs(self):
        prices = np.array(self.price_hist)
        returns = np.diff(prices) / prices[:-1]
        padded_returns = np.zeros(19)
        if len(returns) > 0:
            padded_returns[-len(returns):] = returns

        current_value = self.cash + self.position * self.price_hist[-1]
        obs = np.zeros(26)
        obs[:19] = padded_returns
        obs[19] = self.position / 100.0
        obs[20] = self.cash / 10000.0
        obs[21] = self.spread_mult
        obs[22] = 0.0  # volume imbalance stub
        obs[23] = (current_value - 10000.0) / 10000.0 # portfolio return
        obs[24] = 0.0  # price deviation stub
        obs[25] = self.t / float(self.max_steps)
        return obs

    def _adversary_obs(self):
        return np.array([self.regime_idx, self.drawdown, self.spread_mult])

    def _info(self):
        current_value = self.cash + self.position * self.price_hist[-1]
        return {
            'pnl': current_value - 10000.0,
            'regime': self.REGIMES[self.regime_idx],
            'drawdown': self.drawdown
        }

    class MockPriceProc:
        def __init__(self, e): self.e = e
        @property
        def regime(self): return self.e.REGIMES[self.e.regime_idx]
    
    @property
    def price_proc(self):
        return self.MockPriceProc(self)
