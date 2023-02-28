import numpy as np
import sys, os, pickle
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
    'policy_type'    : 'softmax'
}

ag_config = {
    'alpha_r'        : 1,         # offline learning rate
    'horizon'        : 10,        # planning horizon (minus 1)
    'xi'             : 0.0001,    # EVB replay threshold
    'num_sims'       : 2000,      # number of MC simulations for need
    'sequences'      : True,
    'max_seq_len'    : 8,
    'env_name'       : env,       # gridworld name
}

save_path = '/home/georgy/Documents/Dayan_lab/PhD/bandits/paper/figures/fig3'

def main():

    env_config['barriers'] = [1, 1, 0]

    agent = AgentPOMDP(*[pag_config, ag_config, env_config])
    Q_MB  = agent._solve_mb(1e-5)

    np.save(os.path.join(save_path, 'q_mb.npy'), Q_MB)

    a, b        = 7, 2
    agent.state = 38          # start state
    agent.M     = np.array([[a, b], [0, 1], [1, 0]])
    agent.Q     = Q_MB.copy() # set MF Q values
    Q_history, gain_history, need_history = agent._replay()

    np.save(os.path.join(save_path, 'q_explore_replay.npy'), agent.Q)

    np.save(os.path.join(save_path, 'q_explore_replay_diff.npy'), agent.Q-Q_MB)

    Q              = agent.Q.copy()
    Q_before       = Q.copy()
    Q_after        = Q.copy()
    Q_after[14, 0] = 0.0
    agent.Q        = Q_after.copy()

    np.save(os.path.join(save_path, 'q_explore_online.npy'), agent.Q)

    np.save(os.path.join(save_path, 'q_explore_online_diff.npy'), agent.Q-Q_before)

    Q_before    = agent.Q.copy()

    agent.state = 14
    agent.M     = np.array([[0, 1], [0, 1], [1, 0]])
    Q_history, gain_history, need_history = agent._replay()
    
    np.save(os.path.join(save_path, 'gain_history.npy'), gain_history)
    np.save(os.path.join(save_path, 'need_history.npy'), need_history)

    np.save(os.path.join(save_path, 'q_explore_online_replay.npy'), agent.Q)

    np.save(os.path.join(save_path, 'q_explore_online_replay_diff.npy'), agent.Q-Q_before)

    with open(os.path.join(save_path, 'ag.pkl'), 'wb') as f:
        pickle.dump(agent, f, pickle.HIGHEST_PROTOCOL)

    return None

if __name__ == '__main__':
    main()