import numpy as np
from agent import Agent
import os, shutil

# --- Specify the environment ---
#         0  0  0  0  1
#           -----------
#         0  0  0  0  0
#         0  0  0  0  0

config = np.array([
    [0,  0,  0,  0,  0],
    [0,  0,  0,  0,  0],
    [0,  0,  0,  0,  0]
])

goal_coords  = [0, 4]
start_coords = [2, 3]

rew_value    = 1
config[goal_coords[0], goal_coords[1]] = rew_value

blocked_state_actions = [[1, 1], [2, 1], [3, 1], [4, 1],
                        [6, 0], [7, 0], [8, 0], [9, 0]]

# --- Specify agent parameters ---

# discount factor
gamma = 0.9

# learning rate 
alpha = 0.4

# --- Main function ---
def main():
    agent = Agent(config, start_coords, goal_coords, blocked_state_actions, [1, 2], 0, alpha, gamma)
    agent.run_simulation(num_steps=5000)
    print(agent.Q)
    print(agent.M)
    return None

if __name__ == '__main__':
    main()