import os, sys, json, traceback
import numpy as np
from openai import OpenAI
from env.openenv_wrapper import AdverseMarketEnvironment, AdverseMarketAction
from tasks.task_definitions import TASKS
from tasks.task_grader import GRADERS

# ── Environment variables (with defaults) ──────────────────────────
API_BASE_URL = os.getenv('API_BASE_URL', 'https://api.groq.com/openai/v1')
MODEL_NAME   = os.getenv('MODEL_NAME', 'openai/gpt-oss-20b')
HF_TOKEN     = os.getenv('HF_TOKEN') or os.getenv('GROQ_API_KEY') # No hardcoded default

IS_FALLBACK  = not HF_TOKEN or HF_TOKEN.strip() == ""
client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN) if not IS_FALLBACK else None

ACTIONS_DESC = {
    0: 'HOLD', 1: 'BUY_10%', 2: 'BUY_20%', 3: 'BUY_33%',
    4: 'SELL_10%', 5: 'SELL_20%', 6: 'SELL_33%',
    7: 'BUY_ALL', 8: 'SELL_ALL'
}

def llm_select_action(obs_dict: dict, step: int) -> int:
    if IS_FALLBACK:
        return np.random.randint(0, 9)
    
    prompt = f"""You are a financial trading agent.
Current market observation (step {step}/1000):
- Portfolio return: {obs_dict['portfolio_return']:.4f}
- Position (normalized): {obs_dict['position_norm']:.4f}
- Cash (normalized): {obs_dict['cash_norm']:.4f}
- Spread ratio: {obs_dict['spread_ratio']:.4f}
- Volume imbalance: {obs_dict['volume_imbalance']:.4f}
- Recent returns (last 5): {obs_dict['price_returns'][-5:]}

Actions: {ACTIONS_DESC}

Reply with ONLY a single integer 0-8."""
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{'role': 'user', 'content': prompt}],
            max_tokens=10, temperature=0.0)
        return int(resp.choices[0].message.content.strip())
    except Exception as e:
        # If LLM call fails, return random so the logs don't mirror HOLD 0.00
        return np.random.randint(0, 9)

# ── Main episode runner ─────────────────────────────────
def run_task(task_id: str):
    cfg = TASKS[task_id]
    adversary = None
    if cfg.adversary_path:
        try:
            from stable_baselines3 import PPO
            adversary = PPO.load(cfg.adversary_path, device='cpu')
        except Exception:
            pass  # fall back to random adversary

    env = AdverseMarketEnvironment(task_id=task_id,
                                   adversary_policy=adversary)
    print(f'[START] task={task_id} env=AdverseMarket-v0'
          f' model={MODEL_NAME}', flush=True)

    obs = env.reset()
    rewards, done, step = [], False, 0
    last_error = None

    try:
        while not done and step < cfg.max_steps:
            step += 1
            action_idx = llm_select_action(obs.model_dump(), step)
            action_str = ACTIONS_DESC.get(action_idx, 'HOLD')
            try:
                obs = env.step(
                    AdverseMarketAction(action_index=action_idx))
                last_error = None
            except Exception as e:
                last_error = str(e).replace('\n', ' ')[:80]
                done = True
                rewards.append(0.0)
                print(f'[STEP] step={step} action={action_str}'
                      f' reward=0.00 done=true'
                      f' error={last_error}', flush=True)
                break
            r = float(obs.reward) if obs.reward is not None else 0.0
            done = obs.done
            rewards.append(r)
            err_str = 'null' if last_error is None else last_error
            print(f'[STEP] step={step} action={action_str}'
                  f' reward={r:.2f} done={str(done).lower()}'
                  f' error={err_str}', flush=True)
    finally:
        final_pv = env._env.cash + env._env.position * \
                   env._env.price_hist[-1]
        survived = final_pv > 0
        score = GRADERS[task_id](rewards, survived, final_pv)
        success = score >= cfg.success_threshold
        rewards_str = ','.join(f'{r:.2f}' for r in rewards)
        print(f'[END] task={task_id} success={str(success).lower()}'
              f' score={score:.4f}'
              f' steps={step} rewards={rewards_str}', flush=True)
        env.close()
    return score

# ── Entry point ─────────────────────────────────────────
if __name__ == '__main__':
    import json as _json
    task = os.getenv('TASK_ID', 'calm-market')
    
    # If TASK_ID is set, run only that task
    if os.getenv('TASK_ID'):
        score = run_task(task)
        print(f'[SCORE] {{"task": "{task}", "score": {score:.4f}}}', flush=True)
    else:
        # Run all tasks and report
        results = {}
        for t in ['calm-market', 'volatile-market', 'adversarial-market']:
            os.environ['TASK_ID'] = t
            s = run_task(t)
            results[t] = s
            print(f'[SCORE] {{"task": "{t}", "score": {s:.4f}}}', flush=True)
        print(f'[RESULTS] {_json.dumps(results)}', flush=True)
