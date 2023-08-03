import numpy as np
from belief_tree import Tree
from analysis import plot_root_values
from tex_tree import generate_big_tex_tree
import os, shutil, pickle, sys
# sys.path.join('/home/georgy/Documents/Dayan_lab/PhD/bandits/bandit/code')

# --- Specify parameters ---

# prior belief at the root

# alpha_0, beta_0 = 20, 1
alpha_1, beta_1 = 1, 1

M = {
    0: 0.5, 
    1: np.array([alpha_1, beta_1])
    }

# other parameters
p = {
    'arms':           ['known', 'unknown'],
    'root_belief':    M,
    'init_qvals':     0.6,
    'rand_init':      False,
    'gamma':          0.9,
    'xi':             0.0001,
    'beta':           2,
    'policy_type':    'softmax',
    'sequences':      False,
    'max_seq_len':    None,
    'constrain_seqs': True,
    'horizon':        3
}

# save path
save_folder = '/home/georgy/Documents/Dayan_lab/PhD/bandits/archive/bandit/archive/data/known'

# --- Main function for replay ---
def main(save_path , params, plot_tree=False):
    
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    else: pass
    os.makedirs(save_path)
    
    tree   = Tree(**params)
    qval_history, need_history, replay_history = tree.replay_updates()

    print('Number of replays: %u'%(len(replay_history)-1))
    print('Policy value: %.2f'%tree.evaluate_policy(tree.qval_tree))

    os.mkdir(os.path.join(save_path, 'replay_data'))
    np.save(os.path.join(save_path, 'replay_data', 'qval_history.npy'), qval_history)
    np.save(os.path.join(save_path, 'replay_data', 'need_history.npy'), need_history)
    np.save(os.path.join(save_path, 'replay_data', 'replay_history.npy'), replay_history)

    if plot_tree:
        os.mkdir(os.path.join(save_path, 'tree'))
        for idx in range(len(replay_history)):
            these_replays  = replay_history[:idx+1]
            this_save_path = os.path.join(save_path, 'tree', 'tex_tree_%u.tex'%idx)
            generate_big_tex_tree(tree, these_replays, qval_history[idx], need_history[idx], this_save_path, tree_height=5)

    with open(os.path.join(save_path, 'tree.pkl'), 'wb') as f:
        pickle.dump(tree, f, pickle.HIGHEST_PROTOCOL)
    
    plot_root_values(save_path)

    # save params
    with open(os.path.join(save_path, 'params.txt'), 'w') as f:
        for k, v in p.items():
            f.write(k)
            f.write(':  ')
            f.write(str(v))
            f.write('\n')

    return None

if __name__ == '__main__':
    save_path      = os.path.join(save_folder, '%u'%0)
    main(save_path, p, plot_tree=True)

    # analyse(save_folder)