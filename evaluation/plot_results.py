import os
import matplotlib.pyplot as plt
import numpy as np

def generate_robustness_gap_plot(output_dir):
    """Generates a sample Robustness Gap curve between calm and adversarial markets."""
    os.makedirs(output_dir, exist_ok=True)
    
    steps = np.linspace(0, 500000, 100)
    # Calm market Sharpe ratio asymptotically approaches 2.5
    calm_sharpe = 2.5 - 2.5 * np.exp(-steps / 100000) + np.random.normal(0, 0.05, 100)
    
    # Adversarial market Sharpe ratio peaks low, then agent learns to adapt
    adv_sharpe = 1.0 - 1.0 * np.exp(-steps / 150000) - 0.5 * np.exp(-(steps-250000)**2/1e10) + np.random.normal(0, 0.08, 100)

    plt.figure(figsize=(10, 6))
    plt.plot(steps, calm_sharpe, label="Calm Market (No adversary)", color='blue', linewidth=2)
    plt.plot(steps, adv_sharpe, label="Adversarial Market (Hard)", color='red', linewidth=2)
    
    # Fill between for robustness gap
    plt.fill_between(steps, adv_sharpe, calm_sharpe, color='gray', alpha=0.2, label=r"Robustness Gap ($\Delta$ Sharpe)")
    
    plt.title("Trader Policy Evaluation: Robustness Gap")
    plt.xlabel("Training Steps")
    plt.ylabel("Test Sharpe Ratio")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "robustness_gap.png"), dpi=150)
    plt.close()
    print("Saved robustness_gap.png")

def generate_regime_shift_plot(output_dir):
    """Generates a sample price track illustrating adversary-induced regime shifts."""
    os.makedirs(output_dir, exist_ok=True)
    
    T = 1000
    prices = np.zeros(T)
    prices[0] = 100.0
    
    # Simulated regimes
    # 0: Bull, 1: Bear, 2: Crisis
    regimes = np.zeros(T, dtype=int)
    regimes[0:300] = 0 # Bull
    regimes[300:450] = 2 # Crisis crash by adversary
    regimes[450:800] = 1 # Bear
    regimes[800:1000] = 0 # Bull
    
    np.random.seed(42)
    for t in range(1, T):
        mu = 0.01 if regimes[t] == 0 else (-0.01 if regimes[t] == 1 else -0.05)
        sigma = 0.02 if regimes[t] != 2 else 0.08
        dt = 1.0/252.0
        dW = np.random.normal(0, np.sqrt(dt))
        prices[t] = prices[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW)
        
    plt.figure(figsize=(12, 5))
    
    # Plot price series
    plt.plot(range(T), prices, color='black', linewidth=1.5, label="Mid-Price")
    
    # Shade regimes
    plt.axvspan(0, 300, color='green', alpha=0.1, label='Bull Regime (Adversary Idle)')
    plt.axvspan(300, 450, color='red', alpha=0.3, label='Crisis Injection (Adversary Attack)')
    plt.axvspan(450, 800, color='orange', alpha=0.1, label='Bear Regime')
    plt.axvspan(800, 1000, color='green', alpha=0.1)
    
    plt.title("Adversarial Market Simulation: Regime Tracking")
    plt.xlabel("Simulation Steps")
    plt.ylabel("Asset Price")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "regime_behavior.png"), dpi=150)
    plt.close()
    print("Saved regime_behavior.png")

if __name__ == "__main__":
    generate_robustness_gap_plot('.')
    generate_regime_shift_plot('.')
