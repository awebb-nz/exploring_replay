import numpy as np
import sys, os, pickle, shutil
sys.path.append('/home/georgy/Documents/Dayan_lab/PhD/bandits/paper/code/maze')
from agent_replay import AgentPOMDP
from utils import load_env

np.random.seed(2)

env            = 'tolman123'
env_file_path  = '/home/georgy/Documents/Dayan_lab/PhD/bandits/paper/code/maze/mazes/' + env + '.txt'
env_config     = load_env(env_file_path)

# --- Specify agent parameters ---
pag_config = {
    'alpha'          : 1,
    'beta'           : 2, 
    'need_beta'      : 2,
    'gain_beta'      : 60,          
    'gamma'          : 0.9,
}

ag_config = {
    'alpha_r'        : 1,         # offline learning rate
    'horizon'        : 10,        # planning horizon (minus 1)
    'xi'             : 0.0001,    # EVB replay threshold
    'num_sims'       : 2000,      # number of MC simulations for need
    'sequences'      : False,
    'max_seq_len'    : 8,
    'env_name'       : env,       # gridworld name
}

save_path = '/home/georgy/Documents/Dayan_lab/PhD/bandits/paper/figures/supp/supp5'

def main():

    if os.path.isdir(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)

    priors = [[2, 2], [6, 2], [10, 2], [14, 2], [18, 2], [22, 2]]

    for idx, prior in enumerate(priors):

        this_save_path = os.path.join(save_path, str(idx))
        os.mkdir(this_save_path)

        env_config['barriers'] = [1, 1, 0]

        agent = AgentPOMDP(*[pag_config, ag_config, env_config])
        Q_MB  = agent._solve_mb(1e-5)

        np.save(os.path.join(this_save_path, 'q_mb.npy'), Q_MB)

        a, b        = prior[0], prior[1]
        agent.state = 38          # start state
        agent.M     = np.array([[a, b], [0, 1], [1, 0]])
        agent.Q     = Q_MB.copy() # set MF Q values
        Q_history, gain_history, need_history = agent._replay()

        np.save(os.path.join(this_save_path, 'q_explore_replay.npy'), agent.Q)
        np.save(os.path.join(this_save_path, 'q_explore_replay_diff.npy'), agent.Q-Q_MB)

        print('Done with prior %u'%idx)

        with open(os.path.join(this_save_path, 'ag.pkl'), 'wb') as f:
            pickle.dump(agent, f, pickle.HIGHEST_PROTOCOL)

    return None

if __name__ == '__main__':
    main()