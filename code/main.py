import numpy as np
from belief_tree import Tree
from tex_tree import generate_big_tex_tree
import os, shutil

# --- Specify parameters ---

# prior belief at the root
M = np.array([
    [100, 1],
    [1, 100]
])

# discount factor
gamma = 0.9

# MF Q values at the root
Q = np.array([2, 1], dtype=np.float32)

# planning horizon
horizon = 3

# save path
root_folder = '/home/georgy/Documents/Dayan_lab/PhD/bandits'
save_path   = os.path.join(root_folder, 'data/example_tree/seq/0/')

# --- Main function for replay ---
def main_replay(save_tree=True):
    tree = Tree(M, Q, 1, 'softmax')
    tree.build_tree(horizon)
    qval_history, need_history, replay_history = tree.replay_updates(0.9, 0.0)

    if save_tree:
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        else: pass
        os.makedirs(save_path)

        np.save(os.path.join(save_path, 'qval_history.npy'), qval_history)
        np.save(os.path.join(save_path, 'need_history.npy'), need_history)
        np.save(os.path.join(save_path, 'replay_history.npy'), replay_history)

        for idx, rep in enumerate(replay_history):
            these_replays = replay_history[:idx+1]
            this_save_path = os.path.join(save_path, 'tex_tree_%u.tex'%idx)
            generate_big_tex_tree(horizon, these_replays, qval_history[idx], need_history[idx], this_save_path)

# --- Main function for full Bayesian updates ---
def main_full():
    tree = Tree(M, Q, 1, 'softmax')
    tree.build_tree(horizon)
    tree.full_updates(gamma)

if __name__ == '__main__':
    # main_replay()
    main_full()