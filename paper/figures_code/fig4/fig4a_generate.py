import numpy as np
import sys, os, pickle
sys.path.append('/home/georgy/Documents/Dayan_lab/PhD/bandits/maze/code/multiple_barriers')
from agent_replay import AgentPOMDP
from utils import load_env

np.random.seed(2)

env            = 't'
env_file_path  = '/home/georgy/Documents/Dayan_lab/PhD/bandits/maze/code/mazes/' + env + '.txt'
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
    'alpha_r'        : 1,        # offline learning rate
    'horizon'        : 4,        # planning horizon (minus 1)
    'xi'             : 0.2,      # EVB replay threshold
    'num_sims'       : 2000,     # number of MC simulations for need
    'sequences'      : False,
    'max_seq_len'    : 8,
    'env_name'       : env,      # gridworld name
}

save_path = '/home/georgy/Documents/Dayan_lab/PhD/bandits/paper/figures/fig4/t_maze'

def main():

    env_config['barriers']  = [1]
    env_config['rew_value'] = [0, 0]

    agent = AgentPOMDP(*[pag_config, ag_config, env_config])
    Q_MB  = agent.Q.copy()

    np.save(os.path.join(save_path, 'q_init.npy'), Q_MB)

    a, b        = 7, 2
    agent.state = 7 # start state
    agent.M     = np.array([[a, b]])

    Q_history, gain_history, need_history = agent._replay()

    np.save(os.path.join(save_path, 'q_explore_replay_before.npy'), agent.Q)
    np.save(os.path.join(save_path, 'q_explore_replay_diff_before.npy'), agent.Q-Q_MB)

    agent.rew_value      = [0, 1]
    agent._init_reward()

    Q_history, gain_history, need_history = agent._replay()

    states = [1, 2, 3]
    Q = agent.Q_nans.copy()
    Q_tree = Q_history[-1]
    for hi in reversed(range(agent.horizon)):
        for k, val in Q_tree[hi].items():
            state  = val[0][1]
            if state in states:
                Q_vals = val[1]
                Q[state, :] = Q_vals[state, :]

    np.save(os.path.join(save_path, 'q_explore_replay_after.npy'), Q)
    np.save(os.path.join(save_path, 'q_explore_replay_diff_after.npy'), Q-Q_MB)

    with open(os.path.join(save_path, 'ag.pkl'), 'wb') as f:
        pickle.dump(agent, f, pickle.HIGHEST_PROTOCOL)

    return None

if __name__ == '__main__':
    main()