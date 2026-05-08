# 🧊 Frozen Lake: Reinforcement Learning & Evolutionary Algorithms

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Artificial Intelligence](https://img.shields.io/badge/AI-Machine_Learning-FF6F00?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)

## 📌 Overview

This repository contains a comprehensive solution for the classic **Frozen Lake** environment. The core goal of this project is to navigate an agent across a grid of slippery ice and water holes to reach a target safely. 

To achieve this, the project explores and compares different Artificial Intelligence approaches, specifically focusing on classic **Reinforcement Learning** techniques and **Evolutionary Computing**.

## 🧠 Algorithms Implemented

The project is modularized into different scripts, each implementing a distinct algorithmic approach:

* **`q_Learning.py`**: An off-policy Temporal Difference (TD) control algorithm to find the optimal action-value function independently of the agent's actions.
* **`sarsa.py`**: An on-policy TD control algorithm (State-Action-Reward-State-Action) that learns the value of the policy being executed.
* **`montecarlo.py`**: A method that learns from episodic experience, averaging the returns observed after visits to specific states.
* **`genetic.py`**: An evolutionary approach (Genetic Algorithm) used to evolve a population of potential paths to find the optimal sequence of movements.
* **`agent.py` & `main.py`**: The core environment setup and the main execution script used to run, orchestrate, and compare the models.

## 🚀 How to Run

To test the algorithms locally on your machine, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/LuisCalvetGarcia/Frozen-Lake.git](https://github.com/LuisCalvetGarcia/Frozen-Lake.git)
   cd Frozen-Lake
