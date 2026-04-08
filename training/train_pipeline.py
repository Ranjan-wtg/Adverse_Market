import os
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from env_wrappers import TraderEnv, AdversaryEnv
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from env.adverse_market_env import AdverseMarketEnv

def train_phase_1():
    print("Starting Phase 1: Bootstrapping Adversary...")
    os.makedirs('../checkpoints', exist_ok=True)
    base_env = AdverseMarketEnv()
    env = AdversaryEnv(base_env, trader_policy=None)
    
    model = PPO("MlpPolicy", env, verbose=1, device="cpu")
    model.learn(total_timesteps=50000)
    model.save("../checkpoints/adversary_phase1")
    print("Phase 1 Complete.")

def train_phase_2():
    print("Starting Phase 2: Bootstrapping Trader...")
    adversary = PPO.load("../checkpoints/adversary_phase1")
    base_env = AdverseMarketEnv(adversary_policy=adversary)
    env = TraderEnv(base_env)
    
    model = PPO("MlpPolicy", env, verbose=1, device="cpu")
    model.learn(total_timesteps=50000)
    model.save("../checkpoints/trader_phase2")
    print("Phase 2 Complete.")

def train_phase_3():
    print("Starting Phase 3: Adversarial Co-Training...")
    # Placeholder for alternating logic between SAC and PPO
    print("Phase 3 Complete.")

if __name__ == "__main__":
    train_phase_1()
    train_phase_2()
    train_phase_3()
