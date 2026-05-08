import numpy as np
import random
import gymnasium as gym

from agent import Agent

class AgentSARSA(Agent):
    def __init__(self, alpha, gamma, seed=0):
        super().__init__(long_memoria=0)

        self.__alpha = alpha
        self.__gamma = gamma
        self.q = None
        self.env = None  

        np.random.seed(seed)
        random.seed(seed)


    def epsilon_greedy(self, eps):
        if self.q is None:
            raise ValueError("La Q-table no está inicializada")

        if self.env is None:
            raise ValueError("El entorno (self.env) no está asignado")

        if not hasattr(self, "_state"):
            raise ValueError("self._state no está definido")

        #Exploració
        if random.uniform(0, 1) < eps:
            return int(self.env.action_space.sample())
        #Explotació
        else:
            return int(np.argmax(self.q[self._state]))
        """
        *ANNOTACIÓ:
        El paràmetre eps (epsilon) es redueix progressivament a l'interior de train() 
        perquè l'agent comenci explorant molt (amb eps=1.0) i acabi explotant el coneixement (amb eps=0.01)

        """

    def train(self):
        """
        Entrena l'agent usant SARSA (OnPolicy).
        """
        if self.env is None:
            raise ValueError("Debe asignarse self.env antes de entrenar")

        # Hiperparámetros
        episodes = 15000
        max_steps = 100
        eps = 1.0
        eps_min = 0.01
        eps_decay = 0.999

        n_states = self.env.observation_space.n
        n_actions = self.env.action_space.n
        self.n_actions = n_actions 

        # Inicialización Q-table
        self.q = np.zeros((n_states, n_actions))

        rewards = []

        for _ in range(episodes):

            state, info = self.env.reset()
            self._state = state
            done = False
            total_reward = 0

            # Acción inicial SARSA
            action = self.epsilon_greedy(eps)

            for _ in range(max_steps):

                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                # Preparar estado para siguiente acción
                self._state = next_state
                next_action = self.epsilon_greedy(eps)

                # Actualización SARSA (on-policy)
                td_target = reward + self.__gamma * self.q[next_state, next_action]
                td_error = td_target - self.q[state, action]
                self.q[state, action] += self.__alpha * td_error

                # Avanzar en el episodio
                state = next_state
                action = next_action
                total_reward += reward

                if done:
                    break

            # Decaimiento de epsilon
            eps = max(eps_min, eps * eps_decay)
            rewards.append(total_reward)

            #Obtenir la política final
            policy = np.argmax(self.q, axis=1)

        print("**************************************************************************\n")
        print("                                SARSA                                      \n")
        print("***************************************************************************\n")

        return self.q, policy, rewards


    # Pren l'estat actual i retorna l'acció amb el valor màxim
    def actua(self, estat):
        if self.q is None:
            raise ValueError("El agente debe estar entrenado antes de usar actua().")

        return int(np.argmax(self.q[estat]))
