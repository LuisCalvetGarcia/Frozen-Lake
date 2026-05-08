import numpy as np
import random
import gymnasium as gym

from agent import Agent

class AgentQlearning(Agent):
    """
    Agent que aprèn la política òptima utilitzant l'algorisme Q-Learning (off-policy).
    
    """
    # 1. Constructor: Només accepta alpha i gamma
    def __init__(self, alpha: float, gamma: float, seed: int = 0):
        super().__init__(long_memoria=0)

        # Paràmetres de l'algorisme
        self.__alpha = alpha
        self.__gamma = gamma
        
        self.q = None # Q-table
        self.n_actions = 0
        self.env = None 
        
        np.random.seed(seed)
        random.seed(seed)


    def epsilon_greedy(self, state, eps):
        """Selecciona una acción usando epsilon greedy para el estado actual"""
        # Se requiere n_actions y q para elegir la acción
        if self.q is None or self.n_actions == 0:
             return 0 
        
        if random.uniform(0, 1) < eps:
            # Exploració: acció aleatòria
            return random.randint(0, self.n_actions - 1)
        else:
            # Explotació: acció amb el màxim valor Q
            return int(np.argmax(self.q[state]))


    def train(self):
        """
        Entrena l'agent usant el Q-Learning (Off-Policy).
        La durada i els paràmetres epsilon es defineixen internament.
        """
        if self.env is None:
             raise ValueError("Debe asignarse self.env antes de entrenar")

        # Hiperparàmetres de Durada i Epsilon (Cablejats, coherents amb SARSA)
        episodes = 15000
        max_steps = 100
        eps = 1.0       # Valor local que decau
        eps_min = 0.01
        eps_decay = 0.999

        n_states = self.env.observation_space.n
        self.n_actions = self.env.action_space.n
        
        # Inicialització Q-table
        if self.q is None:
             self.q = np.zeros((n_states, self.n_actions))

        rewards = []

        for _ in range(episodes):
            state, info = self.env.reset()
            done = False
            total_reward = 0

            for _ in range(max_steps):
                
                # 1. Escollir A usant la variable local 'eps'
                action = self.epsilon_greedy(state, eps)

                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                # 2. Calcular el Terme Off-Policy: max_a Q(S', a) (CLAU DE Q-LEARNING)
                if done:
                    best_future_q = 0.0
                else:
                    # En Q-Learning, miramos el mejor valor Q posible en S' (Greedy)
                    best_future_q = np.max(self.q[next_state, :]) 

                # 3. Actualització Q-Learning
                td_target = reward + self.__gamma * best_future_q
                td_error = td_target - self.q[state, action]
                self.q[state, action] += self.__alpha * td_error

                # 4. S <- S'
                state = next_state
                total_reward += reward

                if done:
                    break

            # 5. Decaïment de epsilon 
            eps = max(eps_min, eps * eps_decay)
            rewards.append(total_reward)

        # La política final és el argmax de la Q-table
        policy = np.argmax(self.q, axis=1)
        
        # Prints extra per coherència amb l'original
        print("\n**************************************************************************")
        print("                            Q-Learning                                    ")
        print("***************************************************************************")

        # Retornem els 3 valors (Q-table, Policy, Rewards)
        return self.q, policy, rewards

    def actua(self, estat):
        if self.q is None:
            return 0 
            
        return int(np.argmax(self.q[estat]))