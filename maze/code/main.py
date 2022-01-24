import numpy as np
from agent import Agent
from utils import plot_simulation
import os

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

goal_state   = 4
start_state  = 14

goal_coords  = np.argwhere(np.arange(config.shape[0]*config.shape[1]).reshape(config.shape) == goal_state).flatten()

rew_value    = 1
config[goal_coords[0], goal_coords[1]] = rew_value

blocked_state_actions = [[1, 1], [2, 1], [3, 1], [4, 1],
                        [6, 0], [7, 0], [8, 0], [9, 0]]

# --- Specify agent parameters ---
num_steps = 50
save_path = '/home/georgy/Documents/Dayan_lab/PhD/bandits/maze/data'
# discount factor
gamma     = 0.9 
alpha     = 0.4
horizon   = 3
xi        = 0.1

# --- Main function ---
def main():
    np.random.seed(1234)
    agent      = Agent(config, start_state, goal_state, blocked_state_actions, [1, 2], 0, alpha, gamma, horizon, xi)
    save_data  = os.path.join(save_path, 'moves')
    agent.run_simulation(num_steps=num_steps, save_path=save_data)
    
    save_plots = os.path.join(save_path, 'plots')
    plot_simulation(agent, save_data, save_plots)
    return None

if __name__ == '__main__':
    main()