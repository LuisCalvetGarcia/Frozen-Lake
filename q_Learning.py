"""
Q-Learning Agent Implementation.
This module defines an Off-Policy Temporal Difference (TD) control algorithm 
for reinforcement learning.
"""

import numpy as np
import random
import gymnasium as gym

from agent import Agent

class AgentQlearning(Agent):
    """
    Q-Learning Agent that learns the optimal policy for a Markov Decision Process.
    
    Unlike SARSA (which is On-Policy), Q-Learning is Off-Policy: it updates 
    the Q-value for a state-action pair assuming the greedy (optimal) action 
    will be taken in the next state, regardless of the action actually chosen 
    by the exploration policy.
    """
    
    def __init__(self, alpha: float, gamma: float, seed: int = 0):
        """
        Initializes the Q-Learning Agent.

        Args:
            alpha (float): The learning rate (0 < alpha <= 1).
            gamma (float): The discount factor (0 <= gamma <= 1).
            seed (int): Random seed for reproducibility.
        """
        # Initialize the parent Agent class
        super().__init__(long_memoria=0)

        # Algorithm hyperparameters
        self.__alpha = alpha
        self.__gamma = gamma
        
        self.q = None       # Q-table
        self.n_actions = 0
        self.env = None 
        
        # Set seeds to ensure consistent behavior during evaluation
        np.random.seed(seed)
        random.seed(seed)


    def epsilon_greedy(self, state, eps):
        """
        Selects an action using the epsilon-greedy exploration strategy.

        Args:
            state (int): The current state of the environment.
            eps (float): The probability of choosing a random action (exploration).

        Returns:
            int: The selected action.
        """
        # Guard clause: Ensure environment dimensions and Q-table are initialized
        if self.q is None or self.n_actions == 0:
             return 0 
        
        if random.uniform(0, 1) < eps:
            # Exploration: choose a random action
            return random.randint(0, self.n_actions - 1)
        else:
            # Exploitation: choose the action with the highest Q-value for this state
            return int(np.argmax(self.q[state]))


    def train(self):
        """
        Trains the agent using the Off-Policy Q-Learning algorithm.
        
        The Q-table update rule is:
        Q(S, A) <- Q(S, A) + alpha * [R + gamma * max_a(Q(S', a)) - Q(S, A)]

        Returns:
            tuple: (Final Q-table, Final deterministic policy, List of rewards per episode)
        """
        if self.env is None:
             raise ValueError("Environment (self.env) must be assigned before training.")

        # --- Hyperparameters ---
        # Hardcoded to match SARSA configuration for fair comparison
        episodes = 15000
        max_steps = 100
        
        # Epsilon configuration for the Exploration-Exploitation trade-off
        eps = 1.0       
        eps_min = 0.01
        eps_decay = 0.999

        n_states = self.env.observation_space.n
        self.n_actions = self.env.action_space.n
        
        # Initialize the Q-table with zeros
        if self.q is None:
             self.q = np.zeros((n_states, self.n_actions))

        rewards = []

        # Main training loop
        for _ in range(episodes):
            state, info = self.env.reset()
            done = False
            total_reward = 0

            for _ in range(max_steps):
                
                # 1. Choose action A from state S using the behavior policy (epsilon-greedy)
                action = self.epsilon_greedy(state, eps)

                # Execute action and observe the outcome
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                # 2. Calculate the Off-Policy Term: max_a Q(S', a)
                # THIS IS THE KEY TO Q-LEARNING: We use the max Q-value of the next state
                # to update our current state, assuming optimal future behavior.
                if done:
                    best_future_q = 0.0
                else:
                    best_future_q = np.max(self.q[next_state, :]) 

                # 3. Q-Learning Update
                td_target = reward + self.__gamma * best_future_q
                td_error = td_target - self.q[state, action]
                self.q[state, action] += self.__alpha * td_error

                # 4. Transition to the next state (S <- S')
                state = next_state
                total_reward += reward

                if done:
                    break

            # 5. Decay epsilon at the end of each episode
            eps = max(eps_min, eps * eps_decay)
            rewards.append(total_reward)

        # Extract the final deterministic policy by finding the argmax of each state
        policy = np.argmax(self.q, axis=1)
        
        print("\n**************************************************************************")
        print("                            Q-Learning                                    ")
        print("**************************************************************************")

        return self.q, policy, rewards


    def actua(self, estat):
        """
        Selects the optimal action for a given state based on the learned Q-table.
        Used exclusively during the evaluation/visualization phase (pure exploitation).

        Args:
            estat (int): The current state of the environment.

        Returns:
            int: The greedy action with the highest Q-value.
        """
        if self.q is None:
            return 0 
            
        return int(np.argmax(self.q[estat]))