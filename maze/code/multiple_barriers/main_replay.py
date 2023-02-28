import numpy as np
from agent_replay import AgentPOMDP, AgentMDP
from utils import load_env, plot_simulation
import os, argparse, pickle

# --- set up ---
# parser         = argparse.ArgumentParser()

# parser.add_argument('--environment', '-e', help='specify the environment', type=str)
# parser.add_argument('--nsteps', '-ns', help='number of simulated time steps', type=int)

# args           = parser.parse_args()

# --- Load environment ---
env            = 'tolman123'
env_file_path  = os.path.join('/home/georgy/Documents/Dayan_lab/PhD/bandits/maze/code/mazes', env + '.txt')
env_config     = load_env(env_file_path)

# --- Specify simulation parameters ---
save_path      = os.path.join('/home/georgy/Documents/Dayan_lab/PhD/bandits/maze/data/replay/local/', env, 'no_uncertainty')
num_steps      = 6000

seed           = 0

# --- Specify agent parameters ---
pag_config = {
    'alpha': 1,
    'beta':  10, 
    'gain_beta': 10,
    'need_beta': 10,
    'gamma': 0.9,
    'policy_type': 'softmax',
    'mf_forget': 0.02
}

ag_config = {
    'alpha_r'        : 1,         # offline learning rate
    'horizon'        : 10,        # planning horizon (minus 1)
    'xi'             : 0.001,     # EVB replay threshold
    'num_sims'       : 2000,      # number of MC simulations for need
    'sequences'      : True,
    'max_seq_len'    : 5,        
    'env_name'       : env,       # gridworld name
    'barriers'       : [1, 1, 0]
}

# --- Main function ---
def main():
    np.random.seed(seed)
    # --------------------
    # --- REPLAY AGENT ---
    # -------------------- 
    save_data_path = os.path.join(save_path, str(seed))
    
    # initialise the agent
    agent = AgentPOMDP(*[pag_config, ag_config, env_config])
    # agent   = AgentMDP(*[pag_config, ag_config, env_config])
    agent.M = np.array([[7, 2], [7, 2], [7, 2]])

    # # run the simulation
    agent.run_simulation(num_steps=num_steps, save_path=save_data_path)

    with open(os.path.join(save_data_path, 'ag.pkl'), 'wb') as f:
        pickle.dump(agent, f, pickle.HIGHEST_PROTOCOL)

    # save_plot_path = os.path.join(save_data_path, 'plots')
    # if not os.path.isdir(save_plot_path):
    #     os.mkdir(save_plot_path)

    # plot_simulation(agent, save_data_path, save_plot_path, move_start=70)

    return None

if __name__ == '__main__':
    for seed in range(10):
        main()
        print(seed)