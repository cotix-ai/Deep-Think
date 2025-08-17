import os
import math
import random
import asyncio
import time
from typing import List, Tuple, Any, Dict, Protocol
from dataclasses import dataclass
import openai
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

# =============================================================================
# 1. APPLICATION CONFIGURATION
# =============================================================================

@dataclass
class AppConfig:
    api_base_url: str = os.getenv("LLM_API_BASE", "https://api.openai.com/v1")
    api_key: str = os.getenv("LLM_API_KEY", "YOUR_API_KEY_HERE")
    
    model_policy: str = "YOUR_MODEL"
    model_value: str = "YOUR_MODEL"

    simulations: int = 30
    rollout_depth: int = 4
    c_p: float = 1.5
    k_var: float = 0.7
    lambda_skepticism: float = 0.2

    math_problem: str = "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?"
    task_num_actions: int = 4
    task_max_steps: int = 8

# =============================================================================
# 2. CORE SERVICES AND ABSTRACTIONS
# =============================================================================

class LLMService:
    def __init__(self, api_key: str, base_url: str):
        if not api_key or "YOUR_API_KEY" in api_key:
            raise ValueError("API key is not configured. Please set the MOLE_API_KEY environment variable.")
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def get_chat_completions(self, prompt: str, model: str, n: int, max_tokens: int, temp: float, stop: List[str] = None) -> List[str]:
        try:
            response = await self.client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt}],
                n=n, max_tokens=max_tokens, temperature=temp, stop=stop,
            )
            return [choice.message.content.strip() for choice in response.choices]
        except Exception:
            await asyncio.sleep(2)
            return []

class Task(Protocol):
    async def get_initial_state(self) -> Any: ...
    async def get_possible_actions(self, state: Any) -> List[Tuple[Any, float]]: ...
    async def get_next_state(self, state: Any, action: Any) -> Any: ...
    def is_terminal(self, state: Any) -> bool: ...
    async def get_state_value(self, state: Any) -> float: ...
    async def rollout_policy(self, state: Any) -> Any: ...
    async def format_result(self, state: Any) -> str: ...

# =============================================================================
# 3. MCTS ALGORITHM IMPLEMENTATION (LLM-UCT)
# =============================================================================

class MCTSNode:
    def __init__(self, state: Any, parent: 'MCTSNode' = None, prior_p: float = 0.0, action: Any = None):
        self.state: Any = state
        self.parent: 'MCTSNode' = parent
        self.action: Any = action
        self.children: List['MCTSNode'] = []
        
        self.Q: float = 0.0
        self.Q2: float = 0.0
        self.N: int = 0
        self.prior_p: float = prior_p

    def uct_score_llm(self, C_p: float, k_var: float, lambda_skepticism: float) -> float:
        if self.N == 0:
            return float('inf')

        average_reward = self.Q
        variance = max(0, self.Q2 - (average_reward ** 2))
        pessimistic_q_value = average_reward - k_var * math.sqrt(variance)

        num_siblings = len(self.parent.children) if self.parent else 1
        effective_prior = (1 - lambda_skepticism) * self.prior_p + lambda_skepticism * (1 / num_siblings)
        
        exploration_bonus = C_p * effective_prior * (math.sqrt(self.parent.N) / (1 + self.N))
        
        return pessimistic_q_value + exploration_bonus

    def select_best_child(self, C_p: float, k_var: float, lambda_skepticism: float) -> 'MCTSNode':
        return max(self.children, key=lambda c: c.uct_score_llm(C_p, k_var, lambda_skepticism))

    def expand(self, action_priors: List[Tuple[Any, float]], next_states: List[Any]):
        for i, (action, prob) in enumerate(action_priors):
            self.children.append(MCTSNode(state=next_states[i], parent=self, prior_p=prob, action=action))

    def backpropagate(self, reward: float):
        node = self
        while node is not None:
            node.N += 1
            node.Q += (reward - node.Q) / node.N
            node.Q2 += (reward**2 - node.Q2) / node.N
            node = node.parent

    @property
    def is_fully_expanded(self) -> bool:
        return len(self.children) > 0

class DeepThinker:
    def __init__(self, task: Task, config: AppConfig):
        self.task = task
        self.simulations = config.simulations
        self.rollout_depth = config.rollout_depth
        self.C_p = config.c_p
        self.k_var = config.k_var
        self.lambda_skepticism = config.lambda_skepticism

    async def _select(self, node: MCTSNode) -> MCTSNode:
        while node.is_fully_expanded and not self.task.is_terminal(node.state):
            node = node.select_best_child(self.C_p, self.k_var, self.lambda_skepticism)
        return node
    
    async def _expand(self, node: MCTSNode):
        if self.task.is_terminal(node.state): return
        action_priors = await self.task.get_possible_actions(node.state)
        if not action_priors: return

        next_state_tasks = [self.task.get_next_state(node.state, action) for action, _ in action_priors]
        next_states = await asyncio.gather(*next_state_tasks)
        
        node.expand(action_priors, next_states)

    async def _simulate_and_backpropagate(self, node: MCTSNode, pbar: tqdm_asyncio):
        current_state = node.state
        for _ in range(self.rollout_depth):
            if self.task.is_terminal(current_state): break
            action = await self.task.rollout_policy(current_state)
            if not action: break
            current_state = await self.task.get_next_state(current_state, action)
        
        reward = await self.task.get_state_value(current_state)
        node.backpropagate(reward)
        pbar.update(1)

    async def think(self) -> Any:
        initial_state = await self.task.get_initial_state()
        root = MCTSNode(state=initial_state)

        progress_bar = tqdm_asyncio(
            total=self.simulations, 
            desc="Synthesizing thoughts ", 
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}"
        )

        simulation_tasks = []
        for _ in range(self.simulations):
            leaf_node = await self._select(root)
            await self._expand(leaf_node)
            node_to_simulate = random.choice(leaf_node.children) if leaf_node.children else leaf_node
            simulation_tasks.append(self._simulate_and_backpropagate(node_to_simulate, progress_bar))
        
        await asyncio.gather(*simulation_tasks)
        progress_bar.close()

        if not root.children:
            return root.state
        
        current_node = root
        while not self.task.is_terminal(current_node.state) and current_node.children:
             current_node = max(current_node.children, key=lambda c: c.N)
        return current_node.state

# =============================================================================
# 4. CONCRETE TASK IMPLEMENTATION
# =============================================================================

class SentenceLevelMathTask(Task):
    def __init__(self, problem: str, num_actions: int, max_steps: int, llm_service: LLMService, config: AppConfig):
        self.problem = problem
        self.num_actions = num_actions
        self.max_steps = max_steps
        self.llm_service = llm_service
        self.config = config

    def _build_context(self, state: Dict) -> str:
        context = f"Problem: {self.problem}\n\nLet's think step by step.\n"
        context += "".join(f"Step {i+1}: {step}\n" for i, step in enumerate(state["steps"]))
        return context

    async def get_initial_state(self) -> Dict[str, Any]:
        return {"problem": self.problem, "steps": []}

    async def get_possible_actions(self, state: Dict) -> List[Tuple[str, float]]:
        context = self._build_context(state)
        prompt = f"{context}\nPropose {self.num_actions} diverse and logical next sentences to continue solving the problem."
        
        options = await self.llm_service.get_chat_completions(prompt, self.config.model_policy, n=self.num_actions, max_tokens=70, temp=0.8)
        unique_options = list(dict.fromkeys(opt for opt in options if opt))
        if not unique_options: return []
        
        prob = 1.0 / len(unique_options)
        return [(opt, prob) for opt in unique_options]

    async def get_next_state(self, state: Dict, action: str) -> Dict:
        return {"problem": self.problem, "steps": state["steps"] + [action]}

    def is_terminal(self, state: Dict) -> bool:
        if not state["steps"]: return False
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
# 5. APPLICATION EXECUTION
# =============================================================================

async def main():
    try:
        config = AppConfig()
        
        llm_service = LLMService(api_key=config.api_key, base_url=config.api_base_url)
        
        task = SentenceLevelMathTask(
            problem=config.math_problem,
            num_actions=config.task_num_actions,
            max_steps=config.task_max_steps,
            llm_service=llm_service,
            config=config
        )
        
        thinker = DeepThinker(task=task, config=config)
        
        print("\n" + "=" * 60)
        print("Deep Think // Conceptualizing Pathways...")
        print("=" * 60)

        start_time = time.time()
        final_state = await thinker.think()
        end_time = time.time()
        
        formatted_result = await task.format_result(final_state)

        print("\n" + "=" * 60)
        print("Deep Think // Solution Synthesized.")
        print(f"(Reasoning completed in {end_time - start_time:.2f} seconds)")
        print("=" * 60)
        print(formatted_result)
        print("\n")

    except ValueError as e:
        print(f"\nERROR: {e}")
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user. Exiting.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())
