import numpy as np
from belief_tree import Tree
from analysis import plot_values
from tex_tree import generate_big_tex_tree
import os, shutil, pickle

# --- Specify parameters ---

# prior belief at the root

alpha_0, beta_0 = 4, 2
alpha_1, beta_1 = 2, 2

M = np.array([
    [alpha_0, beta_0],
    [alpha_1, beta_1]
])

# discount factor
gamma = 0.9
xi    = 0.005
beta  = 4

# MF Q values at the root
Q     = np.array([0.0, 0.0])

# planning horizon
horizon = 4

# save path
root_folder = '/home/georgy/Documents/Dayan_lab/PhD/bandits'
save_path   = os.path.join(root_folder, 'rldm/figures/fig1/trees/1')

# --- Main function for replay ---
def main_replay(save_tree=True):
    tree = Tree(M, Q, beta, 'softmax')
    tree.build_tree(horizon)

    qval_history, need_history, replay_history = tree.replay_updates(gamma, xi)
    print(len(replay_history)-1, 'replays', flush=True)
    tree.evaluate_policy(tree.qval_tree)


    if save_tree:
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        else: pass
        os.makedirs(save_path)

        np.save(os.path.join(save_path, 'qval_history.npy'), qval_history)
        np.save(os.path.join(save_path, 'need_history.npy'), need_history)
        np.save(os.path.join(save_path, 'replay_history.npy'), replay_history)

        for idx in range(len(replay_history)):
            these_replays = replay_history[:idx+1]
            this_save_path = os.path.join(save_path, 'tex_tree_%u.tex'%idx)
            generate_big_tex_tree(horizon, these_replays, qval_history[idx], need_history[idx], this_save_path)

        with open(os.path.join(save_path, 'tree.pkl'), 'wb') as f:
            pickle.dump(tree, f, pickle.HIGHEST_PROTOCOL)

        # save params
        with open(os.path.join(save_path, 'params.txt'), 'w') as f:
            f.write('Horizon:       %u\n'%horizon)
            f.write('Prior belief: (alpha_0: %u, beta_0: %u, alpha_1: %u, beta_1: %u)\n'%(alpha_0, beta_0, alpha_1, beta_1))
            f.write('gamma:         %.2f\n'%gamma)
            f.write('xi:            %.4f\n'%xi)
            f.write('beta:          %.2f\n'%beta)
            f.write('MF Q values:  [%.2f, %.2f]\n'%(Q[0], Q[1]))

    plot_values(save_path)

    return None

# --- Main function for full Bayesian updates ---
def main_full():
    tree = Tree(M, Q, beta, 'softmax')
    tree.build_tree(horizon)
    tree.full_updates(gamma)

if __name__ == '__main__':
    main_replay()
    main_full()