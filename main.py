from sarsa import AgentSARSA
from q_Learning import AgentQlearning
from montecarlo import AgentMonteCarlo
from genetic import AgentGenetic

import gymnasium as gym
import numpy as np
import time


"""
Imprimeix la política final de l'agent: l'acció que l'agent prendria
a cada casella si volgués maximitzar la seva recompensa
"""
def print_policy(env, Q):

    # Mapping de acciones del entorno FrozenLake
    arrows = {0: '<', 1: 'v', 2: '>', 3: '^'}

    desc = env.unwrapped.desc.astype("str")   # Mapa original (S, F, H, G)
    grid = []

    for r in range(4):
        row = []
        for c in range(4):
            idx = r * 4 + c

            if desc[r][c] == 'S':
                row.append('S')
            elif desc[r][c] == 'G':
                row.append('G')
            elif desc[r][c] == 'H':
                row.append('H')
            else:
                # Acción greedy aprendida
                a = np.argmax(Q[idx])
                row.append(arrows[a])
        grid.append(row)

    #Visualitzar la grallea de la política aprenguda
    for row in grid:
        print(" ".join(f"{col:>2}" for col in row))
    print()


def main():

    # ===============================================
    # ENTRENAMIENTO (render_mode=None)
    # ===============================================

    #Creació de l'Entorn d'Entrenament
    env = gym.make(
        "FrozenLake-v1",
        is_slippery=True,
        render_mode=None
    )

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    # Crear agente correspondiente dependiendo del caso de pruebas
    agent = AgentSARSA(alpha=0.5, gamma=1.0)
    #agent = AgentQlearning(alpha=0.5, gamma=1.0)
    #agent = AgentMonteCarlo(gamma=1.0)
    #agent = AgentGenetic(pop_size=40, generations=120, mutation_rate=0.1, episodes_eval=40)

    # Asignar entorno al agente
    agent.env = env

    # Entrenar
    Q_table, policy, rewards = agent.train()
    agent.q = Q_table 

    # Validar dimensiones de la tabla Q
    assert agent.q is not None and agent.q.shape == (n_states, n_actions), \
        f"Q-table shape {agent.q.shape if agent.q is not None else None} != ({n_states}, {n_actions})"

    print(f"Entrenamet finalitzat. Episodis: {len(rewards)}, Rendiment dels últims 100 episodis: {np.mean(rewards[-100:]):.3f}")

    print("\nLearned policy:")
    print_policy(env, agent.q)  

    # ===============================================
    # VISUALIZACIÓN (render_mode="human")
    # ===============================================

    # Creació de l'Entorn de Visualització
    env = gym.make(
        "FrozenLake-v1",
        is_slippery=True,
        render_mode="human"
    )

    agent.env = env  # el agente debe usar el nuevo entorno renderizado

    state, info = env.reset() #Iniciar un nou episodi 
    done = False
    total_reward = 0

    # Bucle d'avaluació
    while not done:
        """"
        Acció sense Exploració--> realitza una acció greedy (sense exploració),
                                  utilitzant la política que l'agent va aprendre durant els episodis anteriors
        """
        #Mira directament a la Q-Table (la memòria creada durant els episodis) i retorna l'acció amb el màxim valor Q
        action = agent.actua(state) 

        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        time.sleep(0.2)  # para ver bien la animación

    print(f"Episode visualized, total reward: {total_reward}")
    env.close()


if __name__ == '__main__':
    main()