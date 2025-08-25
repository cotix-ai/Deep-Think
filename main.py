from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time
from typing import List, Tuple, Any, Dict

from mcts import DeepThinker, Task
from llm_service import LLMService
from config import AppConfig


class SentenceLevelMathTask(Task):
    def __init__(self, problem: str, num_actions: int, max_steps: int, llm_service: LLMService, config: AppConfig):
        self.problem = problem
        self.num_actions = num_actions
        self.max_steps = max_steps
        self.llm_service = llm_service
        self.config = config

    def _build_context(self, state: Dict) -> str:
        context = f"Problem: {self.problem}\n\nLet's think step by step.\n"
        context += "".join(f"Step {i+1}: {step}\n" for i,
                           step in enumerate(state["steps"]))
        return context

    async def get_initial_state(self) -> Dict[str, Any]:
        return {"problem": self.problem, "steps": []}

    async def get_possible_actions(self, state: Dict) -> List[Tuple[str, float]]:
        context = self._build_context(state)
        prompt = f"{context}\nPropose {self.num_actions} diverse and logical next sentences to continue solving the problem."

        options = await self.llm_service.get_chat_completions(prompt, self.config.model_policy, n=self.num_actions, max_tokens=70, temp=0.8)
        unique_options = list(dict.fromkeys(opt for opt in options if opt))
        if not unique_options:
            return []

        prob = 1.0 / len(unique_options)
        return [(opt, prob) for opt in unique_options]

    async def get_next_state(self, state: Dict, action: str) -> Dict:
        return {"problem": self.problem, "steps": state["steps"] + [action]}

    def is_terminal(self, state: Dict) -> bool:
        if not state["steps"]:
            return False
        last_step = state["steps"][-1].lower()
        return "the final answer is" in last_step or len(state["steps"]) >= self.max_steps

    async def get_state_value(self, state: Dict) -> float:
        context = self._build_context(state)
        prompt_template = (
            "Evaluate the final answer in the provided solution. Does it correctly solve the problem? "
            "Respond ONLY with a single float from 0.0 (completely wrong) to 1.0 (perfectly correct).\n\n{context}\n\nCorrectness score:"
            if self.is_terminal(state) else
            "How promising is this partial solution path to a correct final answer? Rate its potential "
            "from 0.0 (dead end) to 1.0 (very promising). Respond ONLY with a single float.\n\n{context}\n\nPromising score:"
        )
        prompt = prompt_template.format(context=context)
        results = await self.llm_service.get_chat_completions(prompt, self.config.model_value, n=1, max_tokens=5, temp=0.0)
        try:
            return max(0.0, min(1.0, float(results[0])))
        except (ValueError, IndexError):
            return 0.5 if not self.is_terminal(state) else 0.0

    async def rollout_policy(self, state: Dict) -> str:
        context = self._build_context(state)
        prompt = f"{context}\nWhat is the most direct and logical single next sentence to solve the problem?"
        options = await self.llm_service.get_chat_completions(prompt, self.config.model_policy, n=1, max_tokens=70, temp=0.1)
        return options[0] if options else ""

    async def format_result(self, state: Dict) -> str:
        header = f"Problem:\n{state['problem']}\n"
        solution_steps = "\n".join(f"  - {step}" for step in state['steps'])
        solution = f"\nSolution Path:\n{solution_steps}"
        return header + solution

# =============================================================================
# 5. FastAPI APPLICATION
# =============================================================================


app = FastAPI()


class ThinkRequest(BaseModel):
    problem: str | None = None
    simulations: int | None = None
    rollout_depth: int | None = None
    c_p: float | None = None
    k_var: float | None = None
    lambda_skepticism: float | None = None
    task_num_actions: int | None = None
    task_max_steps: int | None = None
    model_policy: str | None = None
    model_value: str | None = None


@app.post("/think")
async def think_endpoint(request: ThinkRequest):
    try:
        config = AppConfig()

        # Update config with request parameters if provided
        if request.problem:
            config.math_problem = request.problem
        if request.simulations:
            config.simulations = request.simulations
        if request.rollout_depth:
            config.rollout_depth = request.rollout_depth
        if request.c_p:
            config.c_p = request.c_p
        if request.k_var:
            config.k_var = request.k_var
        if request.lambda_skepticism:
            config.lambda_skepticism = request.lambda_skepticism
        if request.task_num_actions:
            config.task_num_actions = request.task_num_actions
        if request.task_max_steps:
            config.task_max_steps = request.task_max_steps
        if request.model_policy:
            config.model_policy = request.model_policy
        if request.model_value:
            config.model_value = request.model_value

        llm_service = LLMService(
            api_key=config.api_key, base_url=config.api_base_url)

        task = SentenceLevelMathTask(
            problem=config.math_problem,
            num_actions=config.task_num_actions,
            max_steps=config.task_max_steps,
            llm_service=llm_service,
            config=config
        )

        thinker = DeepThinker(task=task, config=config)

        start_time = time.time()
        final_state = await thinker.think()
        end_time = time.time()

        formatted_result = await task.format_result(final_state)

        return {
            "status": "success",
            "reasoning_time": f"{end_time - start_time:.2f} seconds",
            "result": formatted_result
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {e}")


@app.get("/")
async def read_root():
    return {"message": "Deep Thinker FastAPI server is running. Use /think endpoint to submit problems."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
