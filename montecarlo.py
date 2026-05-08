import numpy as np
import random
import gymnasium as gym

from agent import Agent 

class AgentMonteCarlo(Agent):
    """
    Agent que aprèn la política òptima utilitzant Monte-Carlo Control (First-Visit).
    
    A diferència de TD, actualitza Q(s,a) utilitzant el Retorn (G) del final de l'episodi.
    """
    def __init__(self, gamma: float , seed: int = 0):
        super().__init__(long_memoria=0)

        # Paràmetres Monte-Carlo
        self.__gamma = gamma
        
        self.q = None # Q-table
        self.n_actions = 0
        self.env = None # Entorn (s'ha d'assignar en main.py)
        
        # Diccionari per emmagatzemar els retorns G per a cada parell (s, a)
        # Clau: (state, action) | Valor: llista de retorns G [G1, G2, ...]
        self.returns = {} 
        
        np.random.seed(seed)
        random.seed(seed)


    def _get_epsilon(self, episode_num: int):
        """Calcula l'epsilon basat en l'episodi (decaïment)."""
        EPS_INITIAL = 1.0
        EPS_MIN = 0.01
        EPS_DECAY_RATE = 0.999 # Més lent que TD perquè MC necessita molta exploració
        
        # Utilitzem l'episodi per fer decaure epsilon
        return max(EPS_MIN, EPS_INITIAL * (EPS_DECAY_RATE ** episode_num))


    def epsilon_greedy(self, state, eps):
        """Selecciona una acció utilitzant la política epsilon-greedy."""
        if self.q is None or self.n_actions == 0:
             return 0 
        
        if random.uniform(0, 1) < eps:
            return random.randint(0, self.n_actions - 1)
        else:
            # Explota la Q-table
            return int(np.argmax(self.q[state]))


    def train(self):
        """
        Entrena l'agent usant Monte-Carlo Control.
        """
        if self.env is None:
             raise ValueError("Debe asignarse self.env antes de entrenar")

        # Hiperparàmetres de Durada 
        episodes = 25000 # MC normalment necessita més episodis que TD per convergir
        max_steps = 100 

        n_states = self.env.observation_space.n
        self.n_actions = self.env.action_space.n
        
        # Inicialització Q-table
        if self.q is None:
             self.q = np.zeros((n_states, self.n_actions))

        rewards = []

        for episode in range(1, episodes + 1):
            
            # Decaïment d'epsilon basat en el número d'episodi
            eps = self._get_epsilon(episode)
            
            # 1. Generar un episodi (Històric de S, A, R)
            episode_data = [] # Emmagatzema tuples (s, a, r)
            state, info = self.env.reset()
            done = False
            total_reward = 0

            for _ in range(max_steps):
                
                # Escollir A (S0, A0) seguint la política π (epsilon-greedy)
                action = self.epsilon_greedy(state, eps)

                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                # Guardar la transició: (Estat, Acció, Recompensa)
                episode_data.append((state, action, reward))

                state = next_state
                total_reward += reward

                if done:
                    break

            rewards.append(total_reward)

            # 2. Avaluació i Millora de la Política (Policy Evaluation & Improvement)
            
            G = 0 # Inicialitzar el Retorn G
            
            # 3. Bucle invers: de T-1 a 0 (Actualització Monte-Carlo)
            # Iterem sobre l'episodi de final a inici (índex: t)
            for t in range(len(episode_data) - 1, -1, -1):
                
                s_t, a_t, r_t = episode_data[t]
                
                # G <- γG + R_t+1 (calcula el Retorn)
                G = self.__gamma * G + r_t 
                
                # First-Visit MC Control: Només actualitzar si és la primera vegada que veiem (s_t, a_t) en l'episodi
                # A Frozen Lake, Every-Visit o First-Visit solen ser suficients
                # Implementem First-Visit: Comprovar si (s_t, a_t) ja ha aparegut en la llista d'estats/accions (0 fins a t-1)
                
                pair = (s_t, a_t)
                
                # Si el parell (s, a) no ha aparegut abans en aquest mateix episodi
                if pair not in [(s, a) for s, a, r in episode_data[:t]]:
                    
                    # Guardar el Retorn (G) a la llista Returns(s, a)
                    if pair not in self.returns:
                        self.returns[pair] = []
                    self.returns[pair].append(G)
                    
                    # Actualitzar Q(s, a) <- average(Returns(s, a))
                    self.q[s_t, a_t] = np.mean(self.returns[pair])
                    
                    # Millora de la Política: π(s) <- argmax_a Q(s, a)
                    # Això passa implícitament en el nostre procés de selecció epsilon-greedy
                    # La Q-table s'actualitza i la pròxima acció (triada per epsilon-greedy) millorarà automàticament.
                    
        # Finalització: La política final és l'argmax de la Q-table
        policy = np.argmax(self.q, axis=1)
        
        print("\n**************************************************************************")
        print("                         Monte-Carlo Control                              ")
        print("***************************************************************************")

        return self.q, policy, rewards

    def actua(self, estat):
        if self.q is None:
            return 0 
            
        return int(np.argmax(self.q[estat]))