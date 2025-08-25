import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class AppConfig:
    api_base_url: str = os.getenv("LLM_API_BASE", "https://api.openai.com/v1")
    api_key: str = os.getenv("LLM_API_KEY", "YOUR_API_KEY_HERE")

    model_policy: str = os.getenv("MODEL_POLICY", "YOUR_MODEL")
    model_value: str = os.getenv("MODEL_VALUE", "YOUR_MODEL")

    simulations: int = int(os.getenv("SIMULATIONS", 30))
    rollout_depth: int = int(os.getenv("ROLLOUT_DEPTH", 4))
    c_p: float = float(os.getenv("C_P", 1.5))
    k_var: float = float(os.getenv("K_VAR", 0.7))
    lambda_skepticism: float = float(os.getenv("LAMBDA_SKEPTICISM", 0.2))

    math_problem: str = os.getenv(
        "MATH_PROBLEM", "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?")
    task_num_actions: int = int(os.getenv("TASK_NUM_ACTIONS", 4))
    task_max_steps: int = int(os.getenv("TASK_MAX_STEPS", 8))
