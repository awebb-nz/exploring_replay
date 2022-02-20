import numpy as np
from agent import Agent
from utils import plot_simulation
import os

# --- Specify the environment --- #
#                                 #
#        0  0  0  0  0  g         #
#        0  0  0  0  0  0         #
#           ----------- X         #
#        0  0  0  0  0  0         #
#        0  0  s  0  0  0         #
#                                 #
# # # # # # # # # # # # # # # # # # 

config       = np.zeros((4, 6))
goal_coords  = [0, 5]
start_coords = [3, 2]
rew_value    = 1

config[goal_coords[0], goal_coords[1]] = rew_value

blocked_state_actions = [
                        [7,  1], [8,  1], [9,  1], [10, 1], [11, 1],
                        [13, 0], [14, 0], [15, 0], [16, 0], [17, 0], 
                        ]

uncertain_states_actions = [17, 0]

# --- Specify simulation parameters ---
#
num_steps  = 5000
save_path  = '/home/georgy/Documents/Dayan_lab/PhD/bandits/maze/data/tmp'
save_data  = os.path.join(save_path, 'moves')
save_plots = os.path.join(save_path, 'plots')

# --- Specify agent parameters ---
#
gamma   = 0.9
alpha   = 1.0
alpha_r = 1.0
horizon = 4 # minus 1
xi      = 1e-2

# --- Main function ---
def main():
    np.random.seed(0)
    # initialise the agent
    agent = Agent(config, start_coords, goal_coords, blocked_state_actions, uncertain_states_actions, alpha, alpha_r, gamma, horizon, xi, policy_temp=4)
    # run the simulation
    agent.run_simulation(num_steps=num_steps, save_path=save_data)
    # plot moves & replays
    plot_simulation(agent, save_data, save_plots)

    return None

if __name__ == '__main__':
    main()