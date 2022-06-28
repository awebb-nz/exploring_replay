import numpy as np
from agent_replay import Agent
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
save_path      = os.path.join('/home/georgy/Documents/Dayan_lab/PhD/bandits/maze/data/replay/local/', env)
num_steps      = 20

seed           = 0

# --- Specify agent parameters ---
ag_config = {
    'alpha'          : 1,         # online learning rate
    'alpha_r'        : 1,         # offline learning rate
    'online_beta'    : 10,        # online inverse temperature
    'gain_beta'      : 50,        # gain inverse temperature
    'need_beta'      : 5,         # need inverse temperature
    'policy_type'    : 'softmax', # policy type [softmax / greedy]
    'gamma'          : 0.4,      # discount factor
    'horizon'        : 4,         # planning horizon (minus 1)
    'xi'             : 1e-9,      # EVB replay threshold
    'num_sims'       : 1000,      # number of MC simulations for need
    'forget'         : 0.2,
    'env_name'       : env        # gridworld name
}

# --- Main function ---
def main():
    np.random.seed(seed)
    # --------------------
    # --- REPLAY AGENT ---
    # --------------------
    save_data_path = os.path.join(save_path, str(seed))
    
    # initialise the agent
    agent = Agent(*[ag_config, env_config])

    # # run the simulation
    # agent.run_simulation(num_steps=num_steps, save_path=save_data_path)

    # with open(os.path.join(save_data_path, 'ag.pkl'), 'wb') as f:
    #     pickle.dump(agent, f, pickle.HIGHEST_PROTOCOL)

    # save_plot_path = os.path.join(save_data_path, 'plots')
    # if not os.path.isdir(save_plot_path):
    #     os.mkdir(save_plot_path)

    # plot_simulation(agent, save_data_path, save_plot_path)
    
    Q_MB        = agent._solve_mb(1e-5, barriers=[0, 1, 0])
    agent.state = 38          # start state
    agent.M     = np.array([[10, 1], [10, 1], [10, 1]])
    agent.Q     = Q_MB.copy() # set MF Q values
    Q_history, gain_history, need_history = agent._replay()

    return None

if __name__ == '__main__':
    main()