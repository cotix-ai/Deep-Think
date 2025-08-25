import math
import random
import asyncio
from typing import List, Tuple, Any, Protocol
from tqdm.asyncio import tqdm_asyncio


class Task(Protocol):
    async def get_initial_state(self) -> Any: ...
    async def get_possible_actions(
        self, state: Any) -> List[Tuple[Any, float]]: ...

    async def get_next_state(self, state: Any, action: Any) -> Any: ...
    def is_terminal(self, state: Any) -> bool: ...
    async def get_state_value(self, state: Any) -> float: ...
    async def rollout_policy(self, state: Any) -> Any: ...
    async def format_result(self, state: Any) -> str: ...


class MCTSNode:
    def __init__(self, state: Any, parent: 'MCTSNode | None' = None, prior_p: float = 0.0, action: Any = None):
        self.state: Any = state
        self.parent: 'MCTSNode | None' = parent
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
        effective_prior = (1 - lambda_skepticism) * \
            self.prior_p + lambda_skepticism * (1 / num_siblings)
        if (self.parent is None):
            raise ValueError("Parent node is None")

        exploration_bonus = C_p * effective_prior * \
            (math.sqrt(self.parent.N) / (1 + self.N))

        return pessimistic_q_value + exploration_bonus

    def select_best_child(self, C_p: float, k_var: float, lambda_skepticism: float) -> 'MCTSNode':
        return max(self.children, key=lambda c: c.uct_score_llm(C_p, k_var, lambda_skepticism))

    def expand(self, action_priors: List[Tuple[Any, float]], next_states: List[Any]):
        for i, (action, prob) in enumerate(action_priors):
            self.children.append(
                MCTSNode(state=next_states[i], parent=self, prior_p=prob, action=action))

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
    def __init__(self, task: Task, config: Any):
        self.task = task
        self.simulations = config.simulations
        self.rollout_depth = config.rollout_depth
        self.C_p = config.c_p
        self.k_var = config.k_var
        self.lambda_skepticism = config.lambda_skepticism

    async def _select(self, node: MCTSNode) -> MCTSNode:
        while node.is_fully_expanded and not self.task.is_terminal(node.state):
            node = node.select_best_child(
                self.C_p, self.k_var, self.lambda_skepticism)
        return node

    async def _expand(self, node: MCTSNode):
        if self.task.is_terminal(node.state):
            return
        action_priors = await self.task.get_possible_actions(node.state)
        if not action_priors:
            return

        next_state_tasks = [self.task.get_next_state(
            node.state, action) for action, _ in action_priors]
        next_states = await asyncio.gather(*next_state_tasks)

        node.expand(action_priors, next_states)

    async def _simulate_and_backpropagate(self, node: MCTSNode, pbar: tqdm_asyncio):
        current_state = node.state
        for _ in range(self.rollout_depth):
            if self.task.is_terminal(current_state):
                break
            action = await self.task.rollout_policy(current_state)
            if not action:
                break
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
            node_to_simulate = random.choice(
                leaf_node.children) if leaf_node.children else leaf_node
            simulation_tasks.append(self._simulate_and_backpropagate(
                node_to_simulate, progress_bar))

        await asyncio.gather(*simulation_tasks)
        progress_bar.close()

        if not root.children:
            return root.state

        current_node = root
        while not self.task.is_terminal(current_node.state) and current_node.children:
            current_node = max(current_node.children, key=lambda c: c.N)
        return current_node.state
