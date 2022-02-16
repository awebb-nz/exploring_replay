import os, pickle
import matplotlib.pyplot as plt
import numpy as np

def plot_values(data_folder, save_fig=True):

    with open(os.path.join(data_folder, 'tree.pkl'), 'rb') as f:
        tree = pickle.load(f)

    qval_history = np.load(os.path.join(data_folder, 'qval_history.npy'), allow_pickle=True)

    root_values   = []
    policy_values = []

    for i in qval_history:
        qval_tree      = i
        policy_values += [tree.evaluate_policy(i)]

        qvals          = qval_tree[0][(0, 0, 0)]
        root_values   += [np.dot(tree._policy(qvals), qvals)]

    # tree.full_updates(tree.gamma)
    # qval_tree = tree.qval_tree
    # qvals     = qval_tree[0][(0, 0, 0)]
    # v_full    = np.max(qvals)
    v_full = 2.86583333

    plt.figure(figsize=(7, 5), dpi=100, constrained_layout=True)
    
    for i in [1, 2]:
        plt.subplot(2, 1, i)

        if i == 1:
            plt.plot(root_values)
            plt.ylabel(r'$V(b_{\rho})$', fontsize=17)
        else:
            plt.plot(policy_values)
            plt.ylabel(r'$V^{\pi}$', fontsize=17)

        plt.axhline(v_full, linestyle='--', color='k', alpha=0.7, label='Optimal value')
        plt.tick_params(axis='y', labelsize=13)

        if i == 1:
            plt.xticks([])
        else:
            plt.xlabel('Number of updates', fontsize=17)
            plt.xticks(range(len(root_values)), range(len(root_values)), fontsize=13)
        
        plt.xlim(0, len(qval_history)-1)
        plt.ylim(0, 3.41258333+0.1)
        plt.legend(prop={'size':13})

    if save_fig:
        plt.savefig(os.path.join(data_folder, 'root_values.png'))
        plt.savefig(os.path.join(data_folder, 'root_values.svg'), transparent=True)