"""
Genetic Algorithm Agent Implementation.
This module defines an Evolutionary Algorithm to solve the environment.
Instead of learning value functions (like TD or MC methods), it directly 
searches the policy space by evolving a population of deterministic policies.
"""

import numpy as np
import random
from agent import Agent

class AgentGenetic(Agent):
    """
    Genetic Algorithm Agent.
    
    Representation:
    Each individual (chromosome) in the population is a deterministic policy:
    an array of size `n_states` where each gene represents an action.
    """

    def __init__(self, pop_size=30, generations=80, mutation_rate=0.15, 
                 episodes_eval=30, seed=0, patience=7): 
        """
        Initializes the Genetic Algorithm Agent.

        Args:
            pop_size (int): Number of individuals (policies) in the population.
            generations (int): Maximum number of evolutionary cycles.
            mutation_rate (float): Probability of a single gene (action) mutating.
            episodes_eval (int): Number of episodes to run to calculate fitness.
            seed (int): Random seed for reproducibility.
            patience (int): Early stopping criterion. Number of generations without 
                            fitness improvement before halting the algorithm.
        """
        super().__init__(long_memoria=0)
        
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.episodes_eval = episodes_eval
        self.patience = patience 
        
        self.env = None
        self.q = None

        np.random.seed(seed)
        random.seed(seed)

    def evaluate_policy(self, policy):
        """
        Fitness Function. 
        Evaluates an individual by running its policy in the environment.
        Because 'is_slippery=True' makes the environment stochastic, we must 
        average the reward over multiple episodes to get a reliable fitness score.

        Args:
            policy (np.ndarray): The policy array to evaluate.

        Returns:
            float: The average reward (fitness) of the policy.
        """
        total_reward = 0
        max_steps = 100 
        
        for _ in range(self.episodes_eval):
            state, info = self.env.reset()
            done = False
            steps = 0
            
            while not done and steps < max_steps: 
                action = policy[state]
                state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                total_reward += reward
                steps += 1
                
        return total_reward / self.episodes_eval

    def tournament_selection(self, population, fitness, k=3):
        """
        Selects a parent for reproduction using Tournament Selection.
        Picks 'k' random individuals and returns the one with the highest fitness.

        Args:
            population (list): The current generation of policies.
            fitness (np.ndarray): The fitness scores of the population.
            k (int): Tournament size. Higher 'k' increases selection pressure.

        Returns:
            np.ndarray: The winning policy chosen to be a parent.
        """
        indices = np.random.choice(len(population), k)
        best = indices[np.argmax(fitness[indices])]
        return population[best]

    def crossover(self, parent1, parent2):
        """
        Performs Uniform Crossover between two parents to create a child.
        Each gene (action) has a 50% chance of being inherited from either parent.

        Args:
            parent1 (np.ndarray): First parent policy.
            parent2 (np.ndarray): Second parent policy.

        Returns:
            np.ndarray: The resulting child policy.
        """
        child = np.zeros_like(parent1)
        for i in range(len(parent1)):
            child[i] = parent1[i] if random.random() < 0.5 else parent2[i]
        return child

    def mutate(self, policy, n_actions):
        """
        Applies Random Resetting Mutation to an individual.
        Helps maintain genetic diversity and prevents premature convergence.

        Args:
            policy (np.ndarray): The policy to mutate.
            n_actions (int): Total number of possible actions (for the new random gene).

        Returns:
            np.ndarray: The mutated policy.
        """
        for i in range(len(policy)):
            if random.random() < self.mutation_rate:
                policy[i] = random.randint(0, n_actions - 1)
        return policy

    def train(self):
        """
        Executes the main Genetic Algorithm loop.
        Stages: Evaluation -> Selection -> Crossover -> Mutation -> Replacement.

        Returns:
            tuple: (Symbolic Q-table, Best Policy, List of best fitness per generation)
        """
        if self.env is None:
            raise ValueError("Environment (self.env) must be assigned before training.")

        n_states = self.env.observation_space.n
        n_actions = self.env.action_space.n

        # Initialize random population
        population = [np.random.randint(0, n_actions, size=n_states) for _ in range(self.pop_size)]
        rewards_best = []
        
        # --- Tracking & Early Stopping Setup ---
        best_fitness_overall = -np.inf 
        generations_without_improvement = 0
        final_best_policy = None 

        for gen in range(self.generations):

            # 1. Evaluate Fitness
            fitness = np.array([self.evaluate_policy(ind) for ind in population])

            current_best_idx = np.argmax(fitness)
            current_best_fitness = fitness[current_best_idx]
            current_best_policy = population[current_best_idx].copy()
            
            # 2. Early Stopping & Elitism Logic
            if current_best_fitness > best_fitness_overall:
                best_fitness_overall = current_best_fitness
                final_best_policy = current_best_policy.copy() 
                generations_without_improvement = 0
                print(f"Gen {gen}: New best fitness: {best_fitness_overall:.2f}")
            else:
                generations_without_improvement += 1
                print(f"Gen {gen}: No improvement. Patience {generations_without_improvement}/{self.patience}")
            
            rewards_best.append(current_best_fitness) 
            
            # Check stopping condition
            if generations_without_improvement >= self.patience:
                print(f"\nFitness hasn't improved in {self.patience} generations. Stopping early.")
                break

            # 3. Reproduction (Create next generation)
            new_population = []
            
            # Elitism: Guarantee the survival of the best individual found so far
            if final_best_policy is not None:
                 new_population.append(final_best_policy.copy()) 
            else:
                 new_population.append(current_best_policy.copy())

            # Fill the rest of the new generation
            while len(new_population) < self.pop_size:
                parent1 = self.tournament_selection(population, fitness)
                parent2 = self.tournament_selection(population, fitness)
                
                child = self.crossover(parent1, parent2)
                child = self.mutate(child, n_actions)
                
                new_population.append(child)

            population = new_population

        # Retrieve the best policy from the entire run
        final_policy = final_best_policy if final_best_policy is not None else population[0]

        # 4. Symbolic Q-Table Generation
        # We create a dummy Q-table filled with 0s, placing a 1.0 at the optimal action 
        # to maintain API compatibility with the RL algorithms (SARSA/Q-Learning)
        # so 'main.py' can visualize the policy without breaking.
        self.q = np.zeros((n_states, n_actions))
        for s in range(n_states):
            self.q[s, final_policy[s]] = 1.0

        print("\n**************************************************************************")
        print("                            GENETIC ALGORITHM                             ")
        print("**************************************************************************")

        return self.q, final_policy, rewards_best

    def actua(self, state):
        """
        Selects the optimal action for a given state based on the learned policy.

        Args:
            state (int): The current state of the environment.

        Returns:
            int: The optimal action.
        """
        if self.q is None:
             return 0 
        return int(np.argmax(self.q[state]))