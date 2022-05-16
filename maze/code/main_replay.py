import numpy as np
from agent_replay import Agent
from utils import load_env, plot_simulation
import os
import pickle

# --- Load environment ---
env_file_path = os.path.join('/home/georgy/Documents/Dayan_lab/PhD/bandits/maze/code/mazes', 'tolman.txt')
env_config    = load_env(env_file_path)

# --- Specify simulation parameters ---
save_path     = '/home/georgy/Documents/Dayan_lab/PhD/bandits/maze/data/replay/tolman_maze'
save_data     = os.path.join(save_path, 'moves')
save_plots    = os.path.join(save_path, 'plots')

num_steps     = 4100
start_replay  = 4000
reset_prior   = True

# --- Specify agent parameters ---
alpha         = 1
alpha_r       = 1
beta          = 10
gamma         = 0.85
horizon       = 3 # minus 1
xi            = 1e-4
num_sims      = 1000

# --- Main function ---
def main():
    np.random.seed(0)
    # --------------------
    # --- REPLAY AGENT ---
    # --------------------

    # initialise the agent
    agent = Agent(alpha, alpha_r, gamma, horizon, xi, num_sims, policy_temp=beta, **env_config)
    # run the simulation
    agent.run_simulation(num_steps=num_steps, start_replay=start_replay, reset_prior=reset_prior, save_path=save_data)
    # save the agent
    # with open(os.path.join(save_path, 'agent.pkl'), 'wb') as ag:
        # pickle.dump(agent, ag, pickle.HIGHEST_PROTOCOL)
    # plot moves & replays
    plot_simulation(agent, save_data, save_plots, start_move=start_replay)
    
    return None

if __name__ == '__main__':
    main()