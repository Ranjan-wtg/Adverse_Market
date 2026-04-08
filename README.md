---
title: AdverseMarket-v0
emoji: 📈
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - finance
  - adversarial-rl
---

# AdverseMarket-v0: Adversarial Market Trading

## Environment Overview & Motivation
This environment models an adversarial financial trading market where the market behaves as a co-evolving RL agent. It challenges standard RL agents by injecting dynamic volatility shocks, regime shifts, and liquidity crises organically driven by the adversary. By placing a trading agent inside a market that actively tries to exploit its weaknesses, this environment tests the **robustness** of trading policies, filling the gap of adversarial evaluation in standard static market scenarios.

## Action Space
The agent has a discrete action space `Discrete(9)` dictating order execution strategies:
- 0: `Hold`
- 1: `Buy 10%`
- 2: `Buy 20%`
- 3: `Buy 33%`
- 4: `Sell 10%`
- 5: `Sell 20%`
- 6: `Sell 33%`
- 7: `Buy All`
- 8: `Sell All`

## Observation Space
The state representation is `Box(26,)` featuring core liquidity and portfolio markers:
- 19 Rolling Price Log-returns
- Position (Normalized)
- Cash (Normalized 10000)
- Spread Ratio 
- Volume Imbalance
- Portfolio Return
- Price Deviation
- Normalized Time Fraction (`step / max_steps`)

## Task Descriptions with Difficulty
1. **calm-market (Easy)**: Trade profitably in a benign bull-market regime. (No trained adversary perturbations.)
2. **volatile-market (Medium)**: Trade profitably and survive under random adversary perturbations.
3. **adversarial-market (Hard)**: Trade profitably against a trained PPO adversarial agent intent on reducing the trader's Sharpe ratio.

## Setup & Usage

**Docker Execution:**
You can spin up this evaluation environment using standard Docker configurations matching the Hugging Face Spaces spec.

```bash
docker build -t adversemarket-v0 .
```

```bash
docker run --rm \
  -e HF_TOKEN=<your_hugging_face_token> \
  -e TASK_ID=calm-market \
  adversemarket-v0
```

*Required Environment Variables:*
- `HF_TOKEN`: Needed to authorize the `OpenAI` client mapping for the evaluation.
- `TASK_ID`: One of `calm-market`, `volatile-market`, or `adversarial-market`.

## Baseline Performance Scores

Below are the quantitative evaluations showcasing the Robustness Gap—highlighting how the `AdverseMarketEnv` stresses baseline Agents effectively.

### Robustness Gap Analysis
![Robustness Gap](robustness_gap.png)

### Simulated Regime Trajectory
![Regime Behavior](regime_behavior.png)

*Final execution scores mapped directly from evaluator outputs:*
- calm-market: 0.85
- volatile-market: 0.61
- adversarial-market: 0.22
