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
    'beta':           20,
    'policy_type':    'softmax',
    'sequences':      False,
    'max_seq_len':    None,
    'constrain_seqs': True,
    'horizon':        4
}

# save path
save_path = '/home/georgy/Documents/Dayan_lab/PhD/bandits/paper/figures/supp/supp3/data'

# --- Main function for replay ---
def main():

    if os.path.isdir(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)
    
    # vary these parameters
    # xis       = np.append(0, np.logspace(np.log2(0.001), np.log2(1.0), 10, base=2))
    horizons  = [3, 4, 5]
    betas     = [1, 2, 4, 'greedy']

    num_trees = 200

    P         = np.zeros((num_trees, len(horizons), len(betas)))
    R         = np.zeros((num_trees, len(horizons), len(betas)))
    nreps     = np.zeros((num_trees, len(horizons), len(betas)), dtype=int)
    R_true    = np.zeros(len(horizons))

    for hidx, horizon in enumerate(horizons):

        for bidx, beta in enumerate(betas):
            p['horizon'] = horizon
            p['beta']    = beta
            # initialise the agent
            tree         = Tree(**p)
            
            # do full bayesian updates
            qval_tree    = tree.full_updates()
            qvals        = qval_tree[0][0]
            v_full       = np.max(qvals)
            R_true[hidx] = v_full
            
            np.random.seed(0)
            for tidx in range(num_trees):
                
                tree = Tree(**p)

                # do replay
                _, _, replays = tree.replay_updates()
                qvals         = tree.qval_tree[0][0]
                v_replay      = tree._value(qvals)

                eval_pol      = tree.evaluate_policy(tree.qval_tree)

                P[tidx, hidx, bidx]     = eval_pol
                R[tidx, hidx, bidx]     = v_replay
                nreps[tidx, hidx, bidx] = len(replays)-1
                
                print('Done with tree %u/%u'%(tidx+1, num_trees))

    file_name = 'noseq'
    np.save(os.path.join(save_path, file_name + '_Rtrue.npy'), R_true)
    np.save(os.path.join(save_path, file_name + '_R.npy'), R)
    np.save(os.path.join(save_path, file_name + '_P.npy'), P)
    np.save(os.path.join(save_path, file_name + '_nreps.npy'), nreps)

    return None

if __name__ == '__main__':
    main()
