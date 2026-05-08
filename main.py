"""
Main execution script for the Frozen Lake environment.
Handles the setup, training, and visual evaluation of various Reinforcement 
Learning and Genetic Algorithm agents.
"""

from sarsa import AgentSARSA
from q_Learning import AgentQlearning
from montecarlo import AgentMonteCarlo
from genetic import AgentGenetic

import gymnasium as gym
import numpy as np
import time


def print_policy(env, Q):
    """
    Visualizes the agent's final learned policy.
    It maps the maximum Q-value for each state to a directional arrow, 
    showing the optimal path the agent intends to take.

    Args:
        env (gym.Env): The initialized Gymnasium environment.
        Q (numpy.ndarray): The Q-table containing state-action values.
    """
    # Mapping actions to visual arrows for the FrozenLake environment
    # 0: Left, 1: Down, 2: Right, 3: Up
    arrows = {0: '<', 1: 'v', 2: '>', 3: '^'}

    # Extract the original map layout (S: Start, F: Frozen, H: Hole, G: Goal)
    desc = env.unwrapped.desc.astype("str")  
    grid = []

    for r in range(4):
        row = []
        for c in range(4):
            idx = r * 4 + c

            # Keep the original environment markers for non-navigable or terminal states
            if desc[r][c] in ['S', 'G', 'H']:
                row.append(desc[r][c])
            else:
                # For standard frozen tiles, append the greedy action learned
                a = np.argmax(Q[idx])
                row.append(arrows[a])
        grid.append(row)

    # Print the resulting policy grid
    for row in grid:
        print(" ".join(f"{col:>2}" for col in row))
    print()


def main():
    """
    Main execution function. 
    Handles the initialization of the environment, agent training, 
    policy extraction, and visual evaluation.
    """
    # ===============================================
    # 1. TRAINING PHASE (render_mode=None for speed)
    # ===============================================

    # Initialize the environment [cite: 27]
    # is_slippery=True introduces stochastic transitions (ice physics) [cite: 29, 30]
    env = gym.make(
        "FrozenLake-v1",
        is_slippery=True, 
        render_mode=None
    )

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    # Instantiate the desired agent. 
    # Uncomment the specific algorithm you wish to train and test.
    agent = AgentSARSA(alpha=0.5, gamma=1.0)
    # agent = AgentQlearning(alpha=0.5, gamma=1.0)
    # agent = AgentMonteCarlo(gamma=1.0)
    # agent = AgentGenetic(pop_size=40, generations=120, mutation_rate=0.1, episodes_eval=40)

    # Link the environment to the agent
    agent.env = env

    # Execute the training loop
    Q_table, policy, rewards = agent.train()
    agent.q = Q_table 

    # Validate that the resulting Q-table matches the environment's state-action space dimensions
    assert agent.q is not None and agent.q.shape == (n_states, n_actions), \
        f"Q-table shape {agent.q.shape if agent.q is not None else None} != ({n_states}, {n_actions})"

    # Display training metrics
    print(f"Training finished. Total episodes: {len(rewards)}")
    print(f"Average reward over the last 100 episodes: {np.mean(rewards[-100:]):.3f}")

    print("\nLearned policy:")
    print_policy(env, agent.q)  

    # ===============================================
    # 2. VISUALIZATION PHASE (render_mode="human")
    # ===============================================

    # Re-initialize the environment with human rendering to visualize the agent's behavior
    env = gym.make(
        "FrozenLake-v1",
        is_slippery=True,
        render_mode="human"
    )

    agent.env = env  # Update the agent's environment reference

    state, info = env.reset() 
    done = False
    total_reward = 0

    # Evaluation loop
    while not done:
        # Exploit the learned policy: choose the greedy action without exploration
        # It queries the Q-Table directly and returns the action with the highest Q-value
        action = agent.actua(state) 

        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        
        # Pause briefly to make the UI animation easily observable
        time.sleep(0.2)  

    print(f"Episode visualized, total reward: {total_reward}")
    env.close()


if __name__ == '__main__':
    main()