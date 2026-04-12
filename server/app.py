"""
FastAPI application for the AdverseMarket-v0 Environment.

Uses openenv-core's create_app() to auto-register all required endpoints:
  /health, /metadata, /schema, /reset, /step, /state, /ws, /mcp

The Gradio dashboard is mounted at /ui.
"""

from openenv.core.env_server.http_server import create_app

from env.openenv_wrapper import (
    AdverseMarketEnvironment,
    AdverseMarketAction,
    AdverseMarketObservation,
)

# Create the OpenEnv-compliant app (auto-registers /health, /schema, etc.)
app = create_app(
    AdverseMarketEnvironment,
    AdverseMarketAction,
    AdverseMarketObservation,
    env_name="AdverseMarket-v0",
    max_concurrent_envs=1,
)

# ── Optional: mount Gradio dashboard at /ui ───────────────────────
import os, sys, subprocess
import gradio as gr


def run_benchmark(task_id):
    env = os.environ.copy()
    env["TASK_ID"] = task_id
    process = subprocess.Popen(
        [sys.executable, "inference.py"],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
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
            run_btn = gr.Button("Run Benchmark", variant="primary")
        with gr.Column():
            output_log = gr.Textbox(label="Execution Logs", lines=20,
                                    max_lines=30, interactive=False)
    run_btn.click(fn=run_benchmark, inputs=[task_select], outputs=output_log)

app = gr.mount_gradio_app(app, demo, path="/ui")


def main(host: str = "0.0.0.0", port: int = 7860):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
