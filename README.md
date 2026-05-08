<h1 align="center">🧊 Frozen Lake: Reinforcement Learning & Evolutionary Algorithms</h1>

<p align="center">
  <a href="https://www.python.org/" target="_blank"><img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"></a>
  <a href="https://en.wikipedia.org/wiki/Reinforcement_learning" target="_blank"><img src="https://img.shields.io/badge/AI-Machine_Learning-FF6F00?style=for-the-badge" alt="AI Machine Learning"></a>
  <a href="https://opensource.org/licenses/MIT" target="_blank"><img src="https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge" alt="License"></a>
</p>

<br> 

<p align="center">
  <img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit1/frozen_lake.gif" alt="Frozen Lake Gameplay" width="350"/>
</p>

<br> 

<p align="center">
  <img src="./screenshots/demo.gif" alt="Frozen Lake Gameplay" width="400"/>
</p>

## 📌 Overview & The Environment

This repository contains a comprehensive solution for the classic **Frozen Lake** environment. The core goal of this project is to navigate an agent across a grid of slippery ice and water holes to reach a target safely without falling in.

[cite_start]In the standard configuration (`is_slippery=True`), the environment is highly stochastic: the agent only moves in the intended direction with a 33.3% probability, while the remaining 66.6% is distributed among perpendicular directions[cite: 26]. [cite_start]Due to these complex and non-deterministic transition dynamics, the project explores and compares different Artificial Intelligence approaches, specifically focusing on **Model-Free Reinforcement Learning** techniques and **Evolutionary Computing**[cite: 4, 35].

The environment is represented by a grid consisting of four types of tiles:
* [cite_start]**`S` (Start):** The safe starting point of our agent[cite: 23].
* **`F` (Frozen):** Solid ice tiles, safe to walk on.
* **`H` (Hole):** Holes in the ice. [cite_start]Stepping here terminates the episode with a reward of 0[cite: 24].
* **`G` (Goal):** The objective. [cite_start]Reaching this tile terminates the episode with a successful reward of +1[cite: 23, 24].

## 🧠 Algorithms Implemented

[cite_start]The project is modularized into different scripts, each implementing a distinct algorithmic approach[cite: 6, 33]:

| Algorithm | Type | Technical Summary |
| :--- | :--- | :--- |
| **`q_Learning.py`** | Off-Policy TD | Learns the optimal action-value function independently of the agent's current policy by assuming greedy future actions. |
| **`sarsa.py`** | On-Policy TD | Updates state-action values based on the specific action taken by the current $\epsilon$-greedy policy, leading to safer, more conservative convergence in risky environments. |
| **`montecarlo.py`** | First-Visit | Estimates the value of state-action pairs by averaging the returns obtained following the first visit to a state at the end of each full episode. |
| **`genetic.py`** | Evolutionary | [cite_start]Evolves a population of discrete deterministic policies through successive generations using tournament selection, uniform crossover, and mutation operators to maximize fitness[cite: 35]. |

## 📊 Experimental Results

[cite_start]The benchmarks below illustrate the performance of each approach when dealing with the sparse reward structure and the stochastic nature of the slippery 4x4 grid[cite: 44].

### Global Performance Comparison

After conducting 5 independent executions per algorithm (25,000 episodes for RL methods, 80 generations for the Genetic Algorithm), the evolutionary approach proved to be vastly superior in this specific environment:

<p align="center">
  <img src="./screenshots/Performance%20Comparison%20Graph.png" alt="Performance Comparison Graph"/>
</p>
<p align="center"><i><b>Figure 1:</b> Performance comparison. The Genetic Algorithm (red) achieves an ~82% success rate in under 20 generations with near-zero variance, significantly outperforming TD methods (SARSA & Q-Learning, ~49%) and Monte Carlo (~37%).</i></p>

## ⚙️ Hyperparameter Analysis & Failure Modes

[cite_start]A critical part of this study was analyzing the sensitivity of these algorithms to their hyperparameters (Learning Rate $\alpha$, Discount Factor $\gamma$, and Exploration Rate $\epsilon$)[cite: 35]. 

For instance, Q-Learning, despite being a powerful algorithm, proved to be highly unstable when configured with an overly aggressive learning rate ($\alpha = 0.9$) in this stochastic environment:

<p align="center">
  <img src="./screenshots/Q-Learning%20Failure%20Analysis.png" alt="Q-Learning Failure Analysis"/>
</p>
<p align="center"><i><b>Figure 2:</b> Learning failure analysis under critical hyperparameters (Q-Learning with $\alpha=0.9$). The off-policy nature combined with aggressive learning leads to severe oscillations and a failure to converge stably.</i></p>

## 🚀 How to Run

To test the algorithms locally on your machine, follow these steps:

1. **Clone the repository:**
  git clone https://github.com/LuisCalvetGarcia/Frozen-Lake.git
2. **Install dependencies:**
  pip install gymnasium numpy
3. **Execute the main script**
  python main.py

## 📄 In-Depth Analysis & Results

For a deep dive into the mathematical foundations, hyperparameters tuning, and a detailed performance comparison between Q-Learning, SARSA, Monte Carlo, and Genetic Algorithms, please refer to the attached project report: Frozen_Lake.pdf.
