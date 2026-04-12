import os, sys, subprocess
import gradio as gr
from fastapi import FastAPI, Body
from env.openenv_wrapper import OpenEnvAdverseMarket, Action, Observation, Reward

# ── FastAPI Server Setup ──────────────────────────────────────────
app = FastAPI(title="AdverseMarket-v0 API")
ENV_INSTANCE = {"instance": None}

def get_env():
    if ENV_INSTANCE["instance"] is None:
        ENV_INSTANCE["instance"] = OpenEnvAdverseMarket(task_id="adversarial-market")
    return ENV_INSTANCE["instance"]

@app.post("/reset", response_model=Observation)
async def reset():
    env = get_env()
    return env.reset()

@app.post("/step")
async def step(action: Action):
    env = get_env()
    obs, reward, done, info = env.step(action)
    # Ensure reward sent to grader is a float, move metadata to info
    info["reward_details"] = reward.model_dump()
    return {
        "observation": obs,
        "reward": float(reward.value),
        "done": done,
        "info": info
    }

@app.get("/state")
async def state():
    env = get_env()
    return env.state()

@app.get("/")
async def root():
    return {"message": "AdverseMarket-v0 API is running. Go to /ui for dashboard."}

# ── Existing Gradio Logic (Modified for Subprocess Context) ───────
def run_benchmark(task_id):
    env = os.environ.copy()
    env["TASK_ID"] = task_id
    process = subprocess.Popen(
        [sys.executable, "inference.py"],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1
    )
    output = ""
    for line in iter(process.stdout.readline, ""):
        output += line
        yield output
    process.stdout.close()
    process.wait()

with gr.Blocks(title="AdverseMarket-v0 Benchmark") as demo:
    gr.Markdown("# AdverseMarket-v0 RL Benchmark")
    gr.Markdown("Testing trading policy robustness against adversarial market regimes.")
    with gr.Row():
        with gr.Column():
            task_select = gr.Dropdown(
                choices=["calm-market", "volatile-market", "adversarial-market"],
                value="calm-market", label="Select Task")
            gr.Markdown("Note: Using hardcoded Groq LLM (openai/gpt-oss-20b) for evaluation.")
            run_btn = gr.Button("Run Benchmark", variant="primary")
        with gr.Column():
            output_log = gr.Textbox(label="Execution Logs", lines=20, max_lines=30, interactive=False)
    run_btn.click(fn=run_benchmark, inputs=[task_select], outputs=output_log)
    gr.Markdown("### Methodology\n1. **Adversary**: PPO agent trained for volatility shocks.\n"
                "2. **Trader**: LLM or Random Baseline.\n3. **API**: Standard OpenEnv endpoints at /reset, /step.")

# ── Mount Gradio to FastAPI ───────────────────────────────────────
app = gr.mount_gradio_app(app, demo, path="/ui")

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
