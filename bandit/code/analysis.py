import numpy as np
from belief_tree import Tree
import os, shutil, pickle
import matplotlib.pyplot as plt

# save path
root_folder = '/home/georgy/Documents/Dayan_lab/PhD/bandits'
save_path   = os.path.join(root_folder, 'rldm/figures/fig1/trees/1')

# --- Main function for replay ---
def main_analysis():
    
    with open(os.path.join(save_path, 'tree.pkl'), 'rb') as f:
        tree = pickle.load(f)

    qval_history = np.load(os.path.join(save_path, 'qval_history.npy'), allow_pickle=True)

    root_values  = []
    for i in qval_history:
        qval_tree = i
        qvals     = tree.evaluate_policy(qval_tree)[(0, 0, 0)]
        root_values += [np.dot(tree._policy(qvals), qvals)]
    
    return None

if __name__ == '__main__':
    main_analysis()