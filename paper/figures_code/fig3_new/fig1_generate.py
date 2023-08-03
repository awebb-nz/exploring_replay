import numpy as np
import sys, os, pickle
sys.path.append('/home/georgy/Documents/Dayan_lab/PhD/bandits/paper/code/maze')
from agent_replay import AgentMDP
from utils import load_env

# --- Load environment ---
env            = 'tolman123_nocheat'
env_file_path  = os.path.join('/home/georgy/Documents/Dayan_lab/PhD/bandits/paper/code/mazes', env + '.txt')
env_config     = load_env(env_file_path)

env_config['barriers'] = [1, 1, 0]

# --- Specify simulation parameters ---
save_path  = os.path.join('/home/georgy/Documents/Dayan_lab/PhD/bandits/paper/figures/fig3_new', 'data', 'yes_forgetting')
num_steps  = 4000

# --- Specify agent parameters ---
pag_config = {
    'alpha'          : 1,
    'beta'           : 10, 
    'gamma'          : 0.9,
    'policy_type'    : 'softmax',
    'mf_forget'      : 0.05
}

ag_config = {
    'alpha_r'        : 1,     # offline learning rate
    'xi'             : 0.001, # EVB replay threshold
    'env_name'       : env,   # gridworld name
}

# --- Main function ---
def main():
    np.random.seed(seed)
    # --------------------
    # --- REPLAY AGENT ---
    # -------------------- 
    save_data_path = os.path.join(save_path, str(seed))
    
    # initialise the agent
    agent = AgentMDP(*[pag_config, ag_config, env_config])

    # # run the simulation
    agent.run_simulation(num_steps=num_steps, save_path=save_data_path)

    with open(os.path.join(save_data_path, 'ag.pkl'), 'wb') as f:
        pickle.dump(agent, f, pickle.HIGHEST_PROTOCOL)

    return None

if __name__ == '__main__':
    for seed in range(10):
        main()
        print(seed)