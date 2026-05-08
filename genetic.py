import numpy as np
import random
from agent import Agent

class AgentGenetic(Agent):

    def __init__(self, pop_size=30, generations=80, mutation_rate=0.15, 
                 episodes_eval=30, seed=0, patience=7): 
        
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
        total_reward = 0
        max_steps = 100 # S'assumeix 100 passos màxims per episodi
        
        for _ in range(self.episodes_eval):
            state, info = self.env.reset()
            done = False
            steps = 0
            
            while not done and steps < max_steps: # Condició de límit de passos
                action = policy[state]
                state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                total_reward += reward
                steps += 1
                
        return total_reward / self.episodes_eval

    def tournament_selection(self, population, fitness, k=3):
        indices = np.random.choice(len(population), k)
        best = indices[np.argmax(fitness[indices])]
        return population[best]

    def crossover(self, parent1, parent2):
        child = np.zeros_like(parent1)
        for i in range(len(parent1)):
            child[i] = parent1[i] if random.random() < 0.5 else parent2[i]
        return child

    def mutate(self, policy, n_actions):
        for i in range(len(policy)):
            if random.random() < self.mutation_rate:
                policy[i] = random.randint(0, n_actions - 1)
        return policy

    def train(self):
        if self.env is None:
            raise ValueError("Debe asignarse self.env antes de entrenar")

        n_states = self.env.observation_space.n
        n_actions = self.env.action_space.n

        population = [np.random.randint(0, n_actions, size=n_states) for _ in range(self.pop_size)]
        rewards_best = []
        
        # --- INICIALITZACIÓ DEL SEGUIMENT ---
        best_fitness_overall = -np.inf 
        generations_without_improvement = 0
        final_best_policy = None 
        # -----------------------------------------------------

        for gen in range(self.generations):

            fitness = np.array([self.evaluate_policy(ind) for ind in population])

            current_best_idx = np.argmax(fitness)
            current_best_fitness = fitness[current_best_idx]
            current_best_policy = population[current_best_idx].copy()
            
            # --- LÒGICA DE EARLY STOPPING I REGISTRE ---
            if current_best_fitness > best_fitness_overall:
                best_fitness_overall = current_best_fitness
                final_best_policy = current_best_policy.copy() 
                generations_without_improvement = 0
                print(f"Gen {gen}: Nueva mejor recompensa: {best_fitness_overall:.2f}")
            else:
                generations_without_improvement += 1
                print(f"Gen {gen}: Sin mejora. Paciencia {generations_without_improvement}/{self.patience}")
            
            rewards_best.append(current_best_fitness) 
            
            # Condició d'Aturada 
            if generations_without_improvement >= self.patience:
                print(f"\nEl fitness no ha mejorado en {self.patience} generaciones--> Se detiene el algoritmo")
                break
            # --------------------------------

            new_population = []
            
            if final_best_policy is not None:
                 new_population.append(final_best_policy.copy()) 
            else:
                 new_population.append(current_best_policy.copy())


            while len(new_population) < self.pop_size:
                parent1 = self.tournament_selection(population, fitness)
                parent2 = self.tournament_selection(population, fitness)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child, n_actions)
                new_population.append(child)

            population = new_population

        # Política final: assegurem que sigui el millor de l'historial
        final_policy = final_best_policy if final_best_policy is not None else population[0]

        # Crear Q-table simbòlica
        self.q = np.zeros((n_states, n_actions))
        for s in range(n_states):
            self.q[s, final_policy[s]] = 1.0

        print("**************************************************************************")
        print("                         GENETIC ALGORITHM")
        print("**************************************************************************")

        return self.q, final_policy, rewards_best


    def actua(self, state):
        return int(np.argmax(self.q[state]))