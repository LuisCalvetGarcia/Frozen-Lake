"""
Monte Carlo Control Agent Implementation.
This module defines a First-Visit Monte Carlo Control algorithm for 
finding the optimal policy in episodic reinforcement learning tasks.
"""

import numpy as np
import random
import gymnasium as gym

from agent import Agent 

class AgentMonteCarlo(Agent):
    """
    Agent that learns the optimal policy using First-Visit Monte Carlo Control.
    
    Unlike Temporal Difference (TD) methods (SARSA, Q-Learning) that update 
    values step-by-step using bootstrapping, Monte Carlo methods learn directly 
    from episodes of experience, updating Q(s,a) using the actual accumulated 
    Return (G) calculated at the end of the episode.
    """
    
    def __init__(self, gamma: float, seed: int = 0):
        """
        Initializes the Monte Carlo Agent.

        Args:
            gamma (float): The discount factor (0 <= gamma <= 1).
            seed (int): Random seed for reproducibility.
        """
        super().__init__(long_memoria=0)

        # Monte Carlo hyperparameters
        self.__gamma = gamma
        
        self.q = None       # Q-table
        self.n_actions = 0
        self.env = None     # Must be assigned via main.py
        
        # Dictionary to store the list of returns for each state-action pair
        # Key: (state, action) tuple | Value: List of historical returns [G1, G2, ...]
        self.returns = {} 
        
        np.random.seed(seed)
        random.seed(seed)


    def _get_epsilon(self, episode_num: int):
        """
        Calculates the decaying epsilon value based on the current episode number.
        Monte Carlo typically requires a slower decay rate than TD methods to ensure 
        adequate exploration across full episodes.

        Args:
            episode_num (int): The current training episode index.

        Returns:
            float: The current epsilon value for the epsilon-greedy policy.
        """
        EPS_INITIAL = 1.0
        EPS_MIN = 0.01
        EPS_DECAY_RATE = 0.999 
        
        return max(EPS_MIN, EPS_INITIAL * (EPS_DECAY_RATE ** episode_num))


    def epsilon_greedy(self, state, eps):
        """
        Selects an action using the epsilon-greedy exploration strategy.

        Args:
            state (int): The current state of the environment.
            eps (float): The probability of choosing a random action (exploration).

        Returns:
            int: The selected action.
        """
        if self.q is None or self.n_actions == 0:
             return 0 
        
        if random.uniform(0, 1) < eps:
            return random.randint(0, self.n_actions - 1)
        else:
            return int(np.argmax(self.q[state]))


    def train(self):
        """
        Trains the agent using First-Visit Monte Carlo Control.
        
        The algorithm consists of two main phases per episode:
        1. Generate a full episode using the current policy (epsilon-greedy).
        2. Iterate backward through the episode to calculate Returns (G) and 
           update the Q-table only for the first occurrence of each (s,a) pair.

        Returns:
            tuple: (Final Q-table, Final deterministic policy, List of rewards per episode)
        """
        if self.env is None:
             raise ValueError("Environment (self.env) must be assigned before training.")

        # --- Hyperparameters ---
        # Monte Carlo generally requires more episodes to converge than TD methods 
        # because updates only happen at the end of an episode, causing higher variance.
        episodes = 25000 
        max_steps = 100 

        n_states = self.env.observation_space.n
        self.n_actions = self.env.action_space.n
        
        # Initialize the Q-table with zeros
        if self.q is None:
             self.q = np.zeros((n_states, self.n_actions))

        rewards = []

        for episode in range(1, episodes + 1):
            
            eps = self._get_epsilon(episode)
            
            # ==========================================
            # Phase 1: Generate an Episode
            # ==========================================
            episode_data = [] # Stores history of transitions as (state, action, reward) tuples
            state, info = self.env.reset()
            done = False
            total_reward = 0

            for _ in range(max_steps):
                
                # Sample an action using the current epsilon-greedy policy
                action = self.epsilon_greedy(state, eps)

                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                # Store the transition
                episode_data.append((state, action, reward))

                state = next_state
                total_reward += reward

                if done:
                    break

            rewards.append(total_reward)

            # ==========================================
            # Phase 2: Policy Evaluation & Improvement
            # ==========================================
            
            G = 0 # Initialize cumulative Return
            
            # Iterate backwards: from T-1 down to 0
            for t in range(len(episode_data) - 1, -1, -1):
                
                s_t, a_t, r_t = episode_data[t]
                
                # Calculate the discounted Return: G <- R_{t+1} + gamma * G
                G = self.__gamma * G + r_t 
                
                pair = (s_t, a_t)
                
                # FIRST-VISIT CHECK:
                # We only update the Q-value if this is the first time the agent 
                # visited this specific state and took this specific action in the current episode.
                # We check this by looking at the history from step 0 to t-1.
                if pair not in [(s, a) for s, a, r in episode_data[:t]]:
                    
                    # Initialize the list for this state-action pair if it doesn't exist
                    if pair not in self.returns:
                        self.returns[pair] = []
                        
                    # Append the computed return
                    self.returns[pair].append(G)
                    
                    # Update Q(s, a) to be the empirical mean of all observed returns
                    self.q[s_t, a_t] = np.mean(self.returns[pair])
                    
                    # Note on Policy Improvement:
                    # Policy improvement happens implicitly. Because the Q-table is updated,
                    # the epsilon_greedy function will automatically select better actions 
                    # in subsequent episodes.
                    
        # Extract the final greedy policy
        policy = np.argmax(self.q, axis=1)
        
        print("\n**************************************************************************")
        print("                         Monte-Carlo Control                              ")
        print("**************************************************************************")

        return self.q, policy, rewards


    def actua(self, estat):
        """
        Selects the optimal action for a given state based on the learned Q-table.
        Used strictly during evaluation/testing.

        Args:
            estat (int): The current state of the environment.

        Returns:
            int: The greedy action with the highest Q-value.
        """
        if self.q is None:
            return 0 
            
        return int(np.argmax(self.q[estat]))