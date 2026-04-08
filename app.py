import os
import gradio as gr
import subprocess
import threading
import queue
import sys

def run_benchmark(task_id):
    env = os.environ.copy()
    env["TASK_ID"] = task_id
    
    process = subprocess.Popen(
        [sys.executable, "inference.py"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
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
                value="calm-market",
                label="Select Task"
            )
            gr.Markdown("Note: Using hardcoded Groq LLM (openai/gpt-oss-20b) for evaluation.")
            run_btn = gr.Button("Run Benchmark", variant="primary")
            
        with gr.Column():
            output_log = gr.Textbox(
                label="Execution Logs",
                lines=20,
                max_lines=30,
                interactive=False
            )
            
    run_btn.click(
        fn=run_benchmark,
        inputs=[task_select],
        outputs=output_log
    )
    
    gr.Markdown("### Methodology")
    gr.Markdown("""
    1. **Adversary**: A PPO agent trained to maximize market volatility and drawdowns.
    2. **Trader**: An LLM-based agent (defaulting to HOLD if no API key is provided) or a trained RL policy.
    3. **Evaluation**: Robustness is measured by the delta in Sharpe ratio between calm and adversarial regimes.
    """)

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)
