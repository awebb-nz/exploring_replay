import numpy as np
import sys, os, shutil, pickle
sys.path.append('/home/georgy/Documents/Dayan_lab/PhD/bandits/bandit/code')
from belief_tree import Tree

# --- Specify parameters ---

# prior belief at the root
alpha_0, beta_0 = 5, 3
alpha_1, beta_1 = 1, 5

M = np.array([
    [alpha_0, beta_0],
    [alpha_1, beta_1]
])

# other parameters
p = {
    'root_belief':    M,
    'rand_init':      True,
    'gamma':          0.9,
    'xi':             0.001,
    'beta':           4,
    'policy_type':    'softmax',
    'sequences':      True,
    'max_seq_len':    None,
    'constrain_seqs': True,
    'horizon':        5
}

# save path
path = '/home/georgy/Documents/Dayan_lab/PhD/bandits/paper/figures/supp/supp4/data'

# --- Main function for replay ---
def main():
    
    num_trees = 200

    seqs      = [True, False]

    for seq in seqs:

        np.random.seed(0)

        for tidx in range(num_trees):

            save_path = os.path.join(path, str(tidx), str(seq), 'replay_data')

            if os.path.isdir(save_path):
                shutil.rmtree(save_path)
            os.makedirs(save_path)

            p['sequences'] = seq
            # initialise the agent
            tree   = Tree(**p)
        
            # do replay
            q_history, n_history, replays = tree.replay_updates()

            np.save(os.path.join(save_path, 'need_history.npy'), n_history)
            np.save(os.path.join(save_path, 'qval_history.npy'), q_history)
            np.save(os.path.join(save_path, 'replay_history.npy'), replays)

            print('Done with tree %u/%u'%(tidx+1, num_trees))

    return None

if __name__ == '__main__':
    main()
