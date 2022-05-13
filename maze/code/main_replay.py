import numpy as np
from agent_replay import Agent
from utils import plot_simulation
import os
import ast
import pickle

# --- Specify the environment --- #
#                                 #
#        0  0  0  0  0  g         #
#        0  0  0  0  0  0         #
#           ----------- X         #
#        0  0  0  0  0  0         #
#        0  0  s  0  0  0         #
#                                 #
# # # # # # # # # # # # # # # # # # 

# Load the environment
env_file_path = os.path.join('/home/georgy/Documents/Dayan_lab/PhD/bandits/maze/code/mazes', 'tolman.txt')
with open(env_file_path, 'r') as f:
    env_config = {}
    for line in f:
        k, v = line.strip().split('=')
        env_config[k.strip()] = ast.literal_eval(v.strip())

# --- Specify simulation parameters ---
#
save_path  = '/home/georgy/Documents/Dayan_lab/PhD/bandits/maze/data/replay/tolman_maze'
save_data  = os.path.join(save_path, 'moves')
save_plots = os.path.join(save_path, 'plots')

# --- Specify agent parameters ---

alpha     = 1
alpha_r   = 1
beta      = 7
gamma     = 0.85
horizon   = 4 # minus 1
xi        = 1e-3
num_sims  = 1000
num_steps = 5050


# prior belief about the barrier
M = np.ones(2)

# --- Main function ---5
def main():
    np.random.seed(0)
    # --------------------
    # --- REPLAY AGENT ---
    # --------------------

    # initialise the agent
    agent = Agent(alpha, alpha_r, gamma, horizon, xi, num_sims, policy_temp=beta, **env_config)
    # run the simulation
    agent.run_simulation(num_steps=num_steps, start_replay=4990, reset_prior=True, save_path=save_data)
    # save the agent
    # with open(os.path.join(save_path, 'agent.pkl'), 'wb') as ag:
        # pickle.dump(agent, ag, pickle.HIGHEST_PROTOCOL)
    # plot moves & replays
    plot_simulation(agent, save_data, save_plots)
    
    return None

if __name__ == '__main__':
    main()