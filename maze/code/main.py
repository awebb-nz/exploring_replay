import numpy as np
from agent import Agent
from utils import plot_simulation
import os

# --- Specify the environment ---
#         0  0  0  0  0  1
#         0  0  0  0  0  0
#           ------------
#         0  0  0  0  X  0
#         0  0  0  0  |  0

config = np.array([
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0]
])

goal_state   = 5
start_state  = 20

goal_coords  = np.argwhere(np.arange(config.shape[0]*config.shape[1]).reshape(config.shape) == goal_state).flatten()

rew_value    = 1
config[goal_coords[0], goal_coords[1]] = rew_value

blocked_state_actions = [[7, 1], [8, 1], [9, 1], [10, 1],
                        [13, 0], [14, 0], [15, 0], [16, 0], 
                        [16, 3], [17, 2], [22, 3], [24, 2]]

# --- Specify simulation parameters ---
num_steps = 150
save_path = '/home/georgy/Documents/Dayan_lab/PhD/bandits/maze/data'

# --- Specify agent parameters ---
gamma     = 0.9
alpha     = 0.4
horizon   = 3
xi        = 0.05

# --- Main function ---
def main():
    np.random.seed(1234)
    agent      = Agent(config, start_state, goal_state, blocked_state_actions, [2, 4], 3, alpha, gamma, horizon, xi, policy_temp=2)
    save_data  = os.path.join(save_path, 'moves')
    agent.run_simulation(num_steps=num_steps, save_path=save_data)
    
    save_plots = os.path.join(save_path, 'plots')
    plot_simulation(agent, save_data, save_plots)

    print(agent.M)
    return None

if __name__ == '__main__':
    main()