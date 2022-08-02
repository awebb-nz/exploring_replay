from logging import root
import numpy as np
from belief_tree import Tree
from analysis import plot_root_values, analyse
from tex_tree import generate_big_tex_tree
import os, shutil, pickle
from copy import deepcopy

# --- Specify parameters ---

# prior belief at the root

alpha_0, beta_0 = 20, 1
alpha_1, beta_1 = 1, 20

M = np.array([
    [alpha_0, beta_0],
    [alpha_1, beta_1]
])

# other parameters
p = {
    'root_belief':    M,
    'init_qvals':     0.6,
    'rand_init':      True,
    'gamma':          0.9,
    'xi':             0.0001,
    'beta':           4,
    'policy_type':    'softmax',
    'sequences':      True,
    'max_seq_len':    None,
    'constrain_seqs': True,
    'horizon':        6
}

# save path
root_folder = '/home/georgy/Documents/Dayan_lab/PhD/bandits/bandit/data/fully_random/trees_horizon6/'

# --- Main function for replay ---
def main(save_path , params, plot_tree=False):
    
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    else: pass
    os.makedirs(save_path)
    
    save_path_seqs = os.path.join(save_path, 'seqs')
    os.mkdir(save_path_seqs)

    params['sequences'] = True
    tree   = Tree(**params)
    q_init = deepcopy(tree.qval_tree)
    qval_history, need_history, replay_history = tree.replay_updates()

    print('Number of replays: %u'%(len(replay_history)-1))
    print('Policy value: %.2f'%tree.evaluate_policy(tree.qval_tree))

    os.mkdir(os.path.join(save_path_seqs, 'replay_data'))
    np.save(os.path.join(save_path_seqs, 'replay_data', 'qval_history.npy'), qval_history)
    np.save(os.path.join(save_path_seqs, 'replay_data', 'need_history.npy'), need_history)
    np.save(os.path.join(save_path_seqs, 'replay_data', 'replay_history.npy'), replay_history)

    if plot_tree:
        os.mkdir(os.path.join(save_path_seqs, 'tree'))
        for idx in range(len(replay_history)):
            these_replays  = replay_history[:idx+1]
            this_save_path = os.path.join(save_path_seqs, 'tree', 'tex_tree_%u.tex'%idx)
            generate_big_tex_tree(tree.horizon, these_replays, qval_history[idx], need_history[idx], this_save_path)

    with open(os.path.join(save_path_seqs, 'tree.pkl'), 'wb') as f:
        pickle.dump(tree, f, pickle.HIGHEST_PROTOCOL)
    
    plot_root_values(save_path_seqs)

    # save params
    with open(os.path.join(save_path_seqs, 'params.txt'), 'w') as f:
        for k, v in p.items():
            f.write(k)
            f.write(':  ')
            f.write(str(v))
            f.write('\n')

    save_path_noseqs = os.path.join(save_path, 'noseqs')
    os.mkdir(save_path_noseqs)
    
    params['sequences'] = False
    tree = Tree(**params)
    tree.qval_tree = q_init
    qval_history, need_history, replay_history = tree.replay_updates()

    print('Number of replays: %u'%(len(replay_history)-1))
    print('Policy value: %.2f'%tree.evaluate_policy(tree.qval_tree))

    os.mkdir(os.path.join(save_path_noseqs, 'replay_data'))
    np.save(os.path.join(save_path_noseqs, 'replay_data', 'qval_history.npy'), qval_history)
    np.save(os.path.join(save_path_noseqs, 'replay_data', 'need_history.npy'), need_history)
    np.save(os.path.join(save_path_noseqs, 'replay_data', 'replay_history.npy'), replay_history)

    if plot_tree:
        os.mkdir(os.path.join(save_path_noseqs, 'tree'))
        for idx in range(len(replay_history)):
            these_replays  = replay_history[:idx+1]
            this_save_path = os.path.join(save_path_noseqs, 'tree', 'tex_tree_%u.tex'%idx)
            generate_big_tex_tree(tree.horizon, these_replays, qval_history[idx], need_history[idx], this_save_path)

    with open(os.path.join(save_path_noseqs, 'tree.pkl'), 'wb') as f:
        pickle.dump(tree, f, pickle.HIGHEST_PROTOCOL)
    
    plot_root_values(save_path_noseqs)

    # save params
    with open(os.path.join(save_path_noseqs, 'params.txt'), 'w') as f:
        for k, v in p.items():
            f.write(k)
            f.write(':  ')
            f.write(str(v))
            f.write('\n')

    return None

if __name__ == '__main__':
    for i in range(200):
        save_path      = os.path.join(root_folder, '%u'%i)
        main(save_path, p, plot_tree=True)

    analyse(root_folder)