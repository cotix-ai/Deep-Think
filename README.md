<div align="center">
  <img src="https://img.shields.io/badge/Framework-DeepThinker-purple?style=for-the-badge&logo=github" alt="Framework Badge">
  <img src="https://img.shields.io/badge/Language-Python-blue?style=for-the-badge&logo=python" alt="Python Badge">
  <img src="https://img.shields.io/badge/Paradigm-LLM--Based%20Reasoning-orange?style=for-the-badge&logo=openai" alt="LLM Reasoning Badge">
  <img src="https://img.shields.io/github/stars/cotix-ai/Deep-Think?style=for-the-badge&color=gold" alt="Stars Badge">
</div>

<br>

<h1 align="center">
  Deep Thinker: An LLM-Powered MCTS Framework
</h1>

<p align="center">
  <i>Bridging the gap between fast, intuitive LLM generation and deep, structured reasoning.</i>
</p>

<br>

>[!IMPORTANT]
> This framework is designed to be a generic, plug-and-play engine for complex reasoning tasks. By defining a simple `Task` interface, you can apply this advanced search-based reasoning to any problem domain, from mathematics and logic puzzles to code generation and strategic planning.

## 🌟 Table of Contents

-   [🌟 Table of Contents](#-table-of-contents)
-   [✨ Introduction](#-introduction)
-   [💡 Core Design Philosophy: From Greedy Generation to Deliberate Exploration](#-core-design-philosophy-from-greedy-generation-to-deliberate-exploration)
-   [🧠 Architecture Core: The Cognitive MCTS Loop](#-architecture-core-the-cognitive-mcts-loop)
-   [🧩 Architectural Components in Detail](#-architectural-components-in-detail)
    -   [The Policy Network (The Proposer)](#the-policy-network-the-proposer)
    -   [The Value Network (The Evaluator)](#the-value-network-the-evaluator)
    -   [The LLM-UCT Algorithm (The Arbiter)](#the-llm-uct-algorithm-the-arbiter)
-   [🔄 How It Works: The Thinking Loop](#-how-it-works-the-thinking-loop)
-   [🚀 Unique Advantages & Innovations](#-unique-advantages--innovations)
-   [🛠️ Getting Started](#️-getting-started)
-   [🤝 Contribution](#-contribution)

<br>

---

## ✨ Introduction

This project introduces **Deep Thinker**, a novel framework that enhances the reasoning capabilities of Large Language Models (LLMs) by integrating them with a powerful search algorithm: **Monte Carlo Tree Search (MCTS)**.

**Deep Thinker** re-conceptualizes LLM-based problem-solving by treating it not as a single, linear generation process, but as a vast **search through a tree of possible thoughts**. It moves beyond the limitations of standard autoregressive models, which often commit to a promising but ultimately incorrect path (a "greedy trap") and struggle to self-correct. The architecture synergizes the generative power of LLMs with the principled exploration of MCTS. This fusion creates a highly robust and deliberate reasoning system capable of exploring many different solution paths, evaluating their potential, and converging on the most promising one, leading to superior final answers.

<br>

---

## 💡 Core Design Philosophy: From Greedy Generation to Deliberate Exploration

**Deep Thinker is not just another prompting technique; it represents a fundamental shift in how we orchestrate LLM inference.** We believe the next leap in AI reasoning requires models that can explore, evaluate, and prune their own lines of thought, much like a human does. It ingeniously translates the abstract "thinking" process into a concrete, state-space search guided by multiple specialized LLM agents.

> "The future of AI reasoning lies in moving from probabilistic text continuation to structured state-space search."

This design aims to surmount the inherent limitations of conventional LLMs in solving multi-step problems where a single early mistake can derail the entire solution, a common failure mode for purely sequential generation methods.

<br>

---

## 🧠 Architecture Core: The Cognitive MCTS Loop

The **Cognitive MCTS Loop** stands as the **bedrock** of the Deep Thinker architecture and serves as the **"Single Source of Truth"** for the entire reasoning process. This mechanism liberates the model from the constraints of a single, irreversible train of thought.

**Functionality:**
The model orchestrates a "team" of LLM-powered agents, each with a distinct role, to navigate the search tree:
1.  **The Policy Network:** Proposes a set of plausible next steps (e.g., tokens, sentences, or actions) from the current state.
2.  **The Value Network:** Evaluates the quality or "promising-ness" of any given state (e.g., a partially completed math solution).
3.  **The Search Algorithm:** Intelligently balances exploring new, uncertain paths (exploration) with deepening the most promising known paths (exploitation).

Every step in the final generated solution is therefore informed not by one impulsive guess, but by thousands of simulated "micro-thoughts" that have explored and evaluated numerous futures.

<br>

---

## 🧩 Architectural Components in Detail

The different components within Deep Thinker fulfill specialized cognitive roles to achieve a holistic reasoning process, driving systemic intelligence through a clear division of labor.

### The Policy Network (The Proposer)
*   **Objective:** To generate a diverse set of high-quality candidate actions from the current state.
*   **Implementation:** In the `TokenLevelMathTask`, this is handled by `get_possible_actions`. It queries an LLM (`MODEL_POLICY`) for the `top_k_actions` next tokens and their probabilities. This provides the raw material for the search, defining "what's possible" from any given point.

### The Value Network (The Evaluator)
*   **Objective:** To provide a reliable heuristic score for any state, guiding the search towards promising areas of the solution space.
*   **Implementation:** The `get_state_value` function queries a different LLM (`MODEL_VALUE`) with a specialized prompt, asking it to score a partial or complete solution on a scale of 0.0 to 1.0. This score is crucial; it acts as the model's "gut feeling" or intuition, telling the search algorithm which paths are worth exploring further.

### The LLM-UCT Algorithm (The Arbiter)
*   **Role:** The heart of the Deep Thinker model, where policy and value information are integrated to make decisions.
*   **Implementation:** The `MCTSNode` class and its `uct_score_llm` method implement a variant of the Upper Confidence bounds for Trees (UCT) algorithm. This formula is the core of the decision-making process. It intelligently combines the `prior_p` (from the Policy Network), the average observed reward `Q` (from the Value Network's feedback), and the visit counts `N` to decide which branch of the thought-tree to explore next.

<br>

---

## 🔄 How It Works: The Thinking Loop

The operation of Deep Thinker follows a clear, iterative cycle, mimicking a structured thought process:

1.  **Selection:** The algorithm starts at the root of the "thought tree" (the initial problem) and traverses down the tree by repeatedly selecting the child node with the highest UCT score. This focuses the search on the most promising known path.
2.  **Expansion:** Once a leaf node (a thought that hasn't been explored yet) is reached, the Policy Network (`get_possible_actions`) is called to generate all possible next steps, expanding the tree.
3.  **Simulation (Rollout):** From one of these new nodes, a fast, lightweight "rollout" policy is used to quickly generate a complete solution. The Value Network (`get_state_value`) then scores this final outcome. This gives a quick estimate of the potential of the newly expanded branch.
4.  **Backpropagation:** The score from the simulation is propagated back up the tree, updating the visit count (`N`) and average reward (`Q`) for every node along the selected path. This refines the "intuition" of the search algorithm for future selections.
5.  **Convergence:** This loop is repeated for a fixed number of `simulations`. The final answer is constructed by following the path from the root with the highest visit count, representing the most thoroughly vetted line of reasoning.

<br>

---

## 🚀 Unique Advantages & Innovations

While advanced prompting techniques like Chain-of-Thought have improved LLM reasoning, they still operate on a single, forward pass. There remains significant scope for improvement in **robustness, error correction, and overcoming initial mistakes.**

**This is precisely the direction that Deep Thinker is designed to profoundly explore and address.**

**Deep Thinker**, through its unique **MCTS-driven, multi-agent LLM architecture**, provides:

*   **Drastically Reduced Reasoning Errors:** By actively exploring and pruning bad solution paths, the model is far less likely to follow a flawed line of logic to its conclusion. The Value Network acts as a constant "reality check."
*   **True "Thinking Outside the Box":** The search algorithm can explore less obvious but potentially correct paths that would be ignored by a greedy, single-pass generation, leading to more creative and robust solutions.
*   **Overcoming Greedy Traps:** The system can "backtrack" from a path that seems promising initially but is later evaluated as a dead end, a capability standard autoregressive models lack.
*   **Tunable Performance:** The trade-off between computation cost and solution quality is made explicit. For harder problems, simply increase the number of `simulations` to allow for deeper "thought" and a higher chance of finding the correct answer.

Through Deep Thinker, we aspire to construct more intelligent, reliable, and performant AI reasoners, transitioning the paradigm from a "stochastic parrot" of text to a true "deliberate thinker" in the problem-solving process.

<br>

---

## 🤝 Contribution

We welcome and encourage contributions to this project! If you have any ideas, suggestions, or discover bugs, please feel free to submit a Pull Request or create an Issue.
