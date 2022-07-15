import numpy as np
from belief_tree import Tree
from analysis import plot_values
from tex_tree import generate_big_tex_tree
import os, shutil, pickle

# --- Specify parameters ---

# prior belief at the root

alpha_0, beta_0 = 14, 10
alpha_1, beta_1 = 4, 3

M = np.array([
    [alpha_0, beta_0],
    [alpha_1, beta_1]
])

# other parameters
p = {
    'root_belief': M,
    'gamma': 0.9,
    'xi': 0.00001,
    'beta': 4,
    'policy_type': 'softmax',
    'sequences': True,
    'max_seq_len': 4,
    'horizon': 5
}

# save path
root_folder = '/home/georgy/Documents/Dayan_lab/PhD/bandits/bandit/data/new/'
save_path   = os.path.join(root_folder, '1')

# --- Main function for replay ---
def main_replay(save_tree=True):
    tree = Tree(**p)
    tree.build_tree()

    qval_history, need_history, replay_history = tree.replay_updates()
    print('Number of replays: %u'%(len(replay_history)-1))
    print('Policy value: %.2f'%tree.evaluate_policy(tree.qval_tree))

    if save_tree:
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        else: pass
        os.makedirs(save_path)

        np.save(os.path.join(save_path, 'qval_history.npy'), qval_history)
        np.save(os.path.join(save_path, 'need_history.npy'), need_history)
        np.save(os.path.join(save_path, 'replay_history.npy'), replay_history)

        for idx in range(len(replay_history)):
            these_replays  = replay_history[:idx+1]
            this_save_path = os.path.join(save_path, 'tex_tree_%u.tex'%idx)
            generate_big_tex_tree(tree.horizon, these_replays, qval_history[idx], need_history[idx], this_save_path)
            # print(these_replays[-1])

        with open(os.path.join(save_path, 'tree.pkl'), 'wb') as f:
            pickle.dump(tree, f, pickle.HIGHEST_PROTOCOL)

        # save params
        with open(os.path.join(save_path, 'params.txt'), 'w') as f:
            f.write('Horizon:       %u\n'%tree.horizon)
            f.write('Prior belief: (alpha_0: %u, beta_0: %u, alpha_1: %u, beta_1: %u)\n'%(alpha_0, beta_0, alpha_1, beta_1))
            f.write('gamma:         %.2f\n'%tree.gamma)
            f.write('xi:            %.4f\n'%tree.xi)
            f.write('beta:          %.2f\n'%tree.beta)
            # f.write('MF Q values:  [%.2f, %.2f]\n'%(Q[0], Q[1]))

    plot_values(save_path)

    return None

# --- Main function for full Bayesian updates ---
def main_full():
    tree = Tree(**p)
    tree.build_tree()
    tree.full_updates()

if __name__ == '__main__':
    main_replay()
    # main_full()