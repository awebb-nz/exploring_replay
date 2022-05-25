import numpy as np
from agent_replay import Agent
from utils import load_env, plot_simulation
import os, argparse, pickle

# --- set up ---
parser         = argparse.ArgumentParser()

parser.add_argument('--environment', '-e', help='specify the environment', type=str)
parser.add_argument('--nsteps', '-ns', help='number of simulated time steps', type=int)

args           = parser.parse_args()

# --- Load environment ---
env_file_path  = os.path.join('/home/georgy/Documents/Dayan_lab/PhD/bandits/maze/code/mazes', args.environment + '.txt')
env_config     = load_env(env_file_path)

# --- Specify simulation parameters ---
save_path      = os.path.join('/home/georgy/Documents/Dayan_lab/PhD/bandits/maze/data/replay/local/', args.environment)
num_steps      = args.nsteps

# --- Specify agent parameters ---
ag_config = {
    'alpha'          : 1,          # online learning rate
    'alpha_r'        : 1,          # offline learning rate
    'online_beta'    : 10,          # inverse temperature
    'gain_beta'      : 50,
    'need_beta'      : 5,
    'policy_type'    : 'softmax',
    'gamma'          : 0.85,        # discount
    'horizon'        : 2,          # minus 1
    'xi'             : 1e-9,       # EVB threshold
    'num_sims'       : 1000,       # number of MC simulations

    'phi'            : 0.5,
    'kappa'          : 0.5,
}

# --- Main function ---
def main():
    np.random.seed(0)
    # --------------------
    # --- REPLAY AGENT ---
    # --------------------
    save_data_path = os.path.join(save_path, str(0))
    
    # initialise the agent
    agent = Agent(*[ag_config, env_config])

    # run the simulation
    agent.run_simulation(num_steps=num_steps, save_path=save_data_path)

    with open(os.path.join(save_data_path, 'ag.pkl'), 'wb') as f:
        pickle.dump(agent, f, pickle.HIGHEST_PROTOCOL)

    save_plot_path = os.path.join(save_data_path, 'plots')
    if not os.path.isdir(save_plot_path):
        os.mkdir(save_plot_path)

    plot_simulation(agent, save_data_path, save_plot_path)

    return None

if __name__ == '__main__':
    main()
