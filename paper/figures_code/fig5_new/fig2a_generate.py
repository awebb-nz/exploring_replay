import numpy as np
import matplotlib.pyplot as plt
import sys, os, shutil
sys.path.append('/home/georgy/Documents/Dayan_lab/PhD/bandits/paper/code/maze')
from agent_replay import AgentPOMDP
from utils import load_env

np.random.seed(2)

env            = 'tolman123'
env_file_path  = '/home/georgy/Documents/Dayan_lab/PhD/bandits/paper/code/mazes/' + env + '.txt'
env_config     = load_env(env_file_path)

# --- Specify agent parameters ---
pag_config = {
    'alpha'          : 1,
    'beta'           : 2, 
    'need_beta'      : 2,
    'gain_beta'      : 60,          
    'gamma'          : 0.9,
    'policy_type'    : 'softmax'
}

ag_config = {
    'alpha_r'        : 1,         # offline learning rate
    'horizon'        : 10,        # planning horizon (minus 1)
    'xi'             : 0.001,     # EVB replay threshold
    'num_sims'       : 2000,      # number of MC simulations for need
    'sequences'      : False,
    'max_seq_len'    : 5,        
    'env_name'       : env,       # gridworld name
}

env_config['barriers'] = [1, 1, 0]

agent = AgentPOMDP(*[pag_config, ag_config, env_config])

save_path = '/home/georgy/Documents/Dayan_lab/PhD/bandits/paper/figures/fig4'

def main():

    if os.path.isdir(save_path):
        shutil.rmtree(save_path)
        os.makedirs(save_path)
    else:
        os.makedirs(save_path)

    Q_MB  = agent._solve_mb(1e-5)

    agent.state  = 38 # start state
    M_range      = [[0, 1], [1, 7], [1, 5], [1, 3], [1, 1], [3, 1], [5, 1], [7, 1], [1, 0]]
    qs           = []

    for M in M_range:
        agent.Q  = Q_MB.copy()
        agent.M  = np.array([[M[0], M[1]], [0, 1], [1, 0]])
        
        _, _, _  = agent._replay()
        
        Q_after  = agent.Q.copy()
        
        qs      += [Q_after[14, :].copy(), Q_after[20, :].copy(), Q_after[19, :].copy(), Q_after[18, :].copy(), Q_after[24, :].copy(), Q_after[30, :].copy(), Q_after[31, :].copy(), Q_after[32, :].copy()]

    np.save(os.path.join(save_path, 'qs_explore.npy'), qs)

    probs = np.zeros(len(M_range))
    for i in range(7, len(qs), 8):
        probs[i//8] = (qs[i][2] == np.nanmax(qs[i]))

    np.save(os.path.join(save_path, 'probas_greedy.npy'), probs)

    return None

if __name__ == '__main__':
    main()