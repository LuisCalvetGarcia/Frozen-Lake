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

## 📌 Overview & The Environment

This repository contains a comprehensive solution for the classic **Frozen Lake** environment. The core goal of this project is to navigate an agent across a grid of slippery ice and water holes to reach a target safely without falling in.

The environment is represented by a grid consisting of four types of tiles:
* **`S` (Start):** The safe starting point of our agent.
* **`F` (Frozen):** Solid ice tiles, safe to walk on.
* **`H` (Hole):** Holes in the ice. Stepping here terminates the episode with a reward of 0.
* **`G` (Goal):** The objective. Reaching this tile terminates the episode with a successful reward of +1.

To master this stochastic environment, the project explores and compares different Artificial Intelligence approaches, specifically focusing on classic **Reinforcement Learning** techniques and **Evolutionary Computing**.

## 🧠 Algorithms Implemented

The project is modularized into different scripts, each implementing a distinct algorithmic approach:

* **`q_Learning.py`**: An off-policy Temporal Difference (TD) control algorithm to find the optimal action-value function independently of the agent's actions.
* **`sarsa.py`**: An on-policy TD control algorithm (State-Action-Reward-State-Action) that learns the value of the policy currently being executed.
* **`montecarlo.py`**: A method that learns from episodic experience, averaging the returns observed after visits to specific states.
* **`genetic.py`**: An evolutionary approach (Genetic Algorithm) used to evolve a population of potential paths to find the optimal sequence of movements.
* **`agent.py` & `main.py`**: The core environment setup and the main execution script used to run, orchestrate, and compare the models.

## 🚀 How to Run

To test the algorithms locally on your machine, follow these steps:

1. **Clone the repository:**
  git clone https://github.com/LuisCalvetGarcia/Frozen-Lake.git
2. **Install dependencies:**
  pip install gymnasium numpy
3. **Execute the main script**
  python main.py

## 📄 In-Depth Analysis & Results

For a deep dive into the mathematical foundations, hyperparameters tuning, and a detailed performance comparison between Q-Learning, SARSA, Monte Carlo, and Genetic Algorithms, please refer to the attached project report: IA_P2.pdf.
