from dataclasses import dataclass
from typing import Optional

@dataclass
class TaskConfig:
    task_id: str
    difficulty: str
    description: str
    adversary_path: Optional[str]   # None = random adversary
    max_steps: int
    success_threshold: float

TASKS = {
    'calm-market': TaskConfig(
        task_id='calm-market', difficulty='easy',
        description='Trade in benign bull market, no learned adversary.',
        adversary_path=None, max_steps=1000, success_threshold=0.5),
    'volatile-market': TaskConfig(
        task_id='volatile-market', difficulty='medium',
        description='Survive random adversary perturbations.',
        adversary_path=None, max_steps=1000, success_threshold=0.6),
    'adversarial-market': TaskConfig(
        task_id='adversarial-market', difficulty='hard',
        description='Trade against trained PPO adversary.',
        adversary_path='checkpoints/adversary_phase1',
        max_steps=1000, success_threshold=0.7),
}
