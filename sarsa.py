"""
SARSA (State-Action-Reward-State-Action) Agent Implementation.
This module defines an On-Policy Temporal Difference (TD) control algorithm 
for reinforcement learning.
"""

import numpy as np
import random
import gymnasium as gym

from agent import Agent


class AgentSARSA(Agent):
    """
    SARSA Agent that learns to solve Markov Decision Processes (MDPs).
    Being an On-Policy algorithm, it updates the Q-values based on the action 
    actually taken by the current policy (epsilon-greedy).
    """

    def __init__(self, alpha, gamma, seed=0):
        """
        Initializes the SARSA Agent.

        Args:
            alpha (float): The learning rate (0 < alpha <= 1). Determines to what extent 
                           newly acquired information overrides old information.
            gamma (float): The discount factor (0 <= gamma <= 1). Determines the importance 
                           of future rewards.
            seed (int): Random seed for reproducibility.
        """
        # Initialize the parent Agent class (assuming memory isn't needed for pure tabular SARSA)
        super().__init__(long_memoria=0)

        self.__alpha = alpha
        self.__gamma = gamma
        self.q = None
        self.env = None  

        # Set seeds for deterministic behavior during debugging/evaluation
        np.random.seed(seed)
        random.seed(seed)


    def epsilon_greedy(self, eps):
        """
        Selects an action using the epsilon-greedy exploration strategy.

        Args:
            eps (float): The probability of choosing a random action (exploration).

        Returns:
            int: The selected action.
        """
        if self.q is None:
            raise ValueError("Q-table is not initialized.")

        if self.env is None:
            raise ValueError("Environment (self.env) is not assigned.")

        if not hasattr(self, "_state"):
            raise ValueError("Current state (self._state) is not defined.")

        # Exploration: choose a random action from the action space
        if random.uniform(0, 1) < eps:
            return int(self.env.action_space.sample())
        
        # Exploitation: choose the action with the highest estimated Q-value for the current state
        else:
            return int(np.argmax(self.q[self._state]))


    def train(self):
        """
        Trains the agent using the SARSA algorithm.
        
        The Q-table update rule is:
        Q(S, A) <- Q(S, A) + alpha * [R + gamma * Q(S', A') - Q(S, A)]

        Returns:
            tuple: (Final Q-table, Final deterministic policy, List of rewards per episode)
        """
        if self.env is None:
            raise ValueError("Environment must be assigned before training.")

        # --- Hyperparameters ---
        episodes = 15000
        max_steps = 100
        
        # Epsilon configuration for the Exploration-Exploitation trade-off
        # Starts at 1.0 (100% exploration) and decays to 0.01 (heavy exploitation)
        eps = 1.0
        eps_min = 0.01
        eps_decay = 0.999

        n_states = self.env.observation_space.n
        n_actions = self.env.action_space.n
        self.n_actions = n_actions 

        # Initialize the Q-table with zeros
        self.q = np.zeros((n_states, n_actions))
        rewards = []

        # Main training loop
        for _ in range(episodes):

            state, info = self.env.reset()
            self._state = state
            done = False
            total_reward = 0

            # In SARSA, the initial action must be selected BEFORE the step loop begins
            action = self.epsilon_greedy(eps)

            for _ in range(max_steps):

                # Execute the action and observe the environment's response
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                # Update the internal state reference and select the NEXT action (A')
                self._state = next_state
                next_action = self.epsilon_greedy(eps)

                # SARSA Update (On-Policy)
                # We use the action 'next_action' that the policy actually intends to take
                td_target = reward + self.__gamma * self.q[next_state, next_action]
                td_error = td_target - self.q[state, action]
                self.q[state, action] += self.__alpha * td_error

                # Transition to the next step
                state = next_state
                action = next_action
                total_reward += reward

                if done:
                    break

            # Decay epsilon after each episode to gradually shift from exploration to exploitation
            eps = max(eps_min, eps * eps_decay)
            rewards.append(total_reward)

        # Extract the final deterministic policy (the best action for each state)
        policy = np.argmax(self.q, axis=1)

        print("**************************************************************************\n")
        print("                                SARSA                                     \n")
        print("**************************************************************************\n")

        return self.q, policy, rewards


    def actua(self, estat):
        """
        Selects the optimal action for a given state based on the learned Q-table.
        Used strictly during evaluation/testing (pure exploitation, no exploration).

        Args:
            estat (int): The current state of the environment.

        Returns:
            int: The greedy action with the highest Q-value.
        """
        if self.q is None:
            raise ValueError("The agent must be trained before calling actua().")

        return int(np.argmax(self.q[estat]))