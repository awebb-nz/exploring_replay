import numpy as np
import sys, os, pickle
sys.path.append('/home/georgy/Documents/Dayan_lab/PhD/bandits/maze/code/multiple_barriers')
from agent_replay import AgentPOMDP
from utils import load_env

np.random.seed(2)

env            = 'tolman1234'
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
    'horizon'        : 6,       # planning horizon (minus 1)
    'xi'             : 0.001,    # EVB replay threshold
    'num_sims'       : 2000,     # number of MC simulations for need
    'sequences'      : True,
    'max_seq_len'    : 4,
    'env_name'       : env,      # gridworld name
}

save_path = '/home/georgy/Documents/Dayan_lab/PhD/bandits/paper/figures/fig4'

def main():

    env_config['barriers'] = [1, 1, 1, 1]

    agent = AgentPOMDP(*[pag_config, ag_config, env_config])
    Q_MB  = agent._solve_mb(1e-5)

    np.save(os.path.join(save_path, 'q_mb.npy'), Q_MB)

    a1, b1      = 7, 2
    a2, b2      = 7, 2
    agent.state = 38          # start state
    agent.M     = np.array([[0, 1], [a1, b1], [0, 1], [a2, b2]])
    agent.Q     = Q_MB.copy() # set MF Q values
    Q_history, gain_history, need_history = agent._replay()

    belief_tree = Q_history[-1]

    Q1 = agent.Q.copy()
    for hi in range(agent.horizon):
        for k, v in belief_tree[hi].items():
            if np.array_equal(agent.M, v[0][0]):
                s = v[0][1]
                q = v[1]
                Q1[s, :] = q[s, :].copy()

    new_M = agent.M.copy()
    new_M[3, :] = [1, 0]
    Q2 = agent.Q.copy()
    for hi in range(agent.horizon):
        for k, v in belief_tree[hi].items():
            if np.array_equal(new_M, v[0][0]):
                s = v[0][1]
                q = v[1]
                Q2[s, :] = q[s, :].copy()

    np.save(os.path.join(save_path, 'q_explore_replay1.npy'), Q1)
    np.save(os.path.join(save_path, 'q_explore_replay2.npy'), Q2)
    np.save(os.path.join(save_path, 'q_explore_replay_diff.npy'), agent.Q-Q_MB)

    with open(os.path.join(save_path, 'ag.pkl'), 'wb') as f:
        pickle.dump(agent, f, pickle.HIGHEST_PROTOCOL)

    return None

if __name__ == '__main__':
    main()