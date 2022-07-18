import os, pickle
from belief_tree import Tree
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy

def plot_root_values(data_folder):

    with open(os.path.join(data_folder, 'tree.pkl'), 'rb') as f:
        tree = pickle.load(f)

    qval_history = np.load(os.path.join(data_folder, 'qval_history.npy'), allow_pickle=True)

    root_values   = []
    policy_values = []

    for i in qval_history:
        qval_tree      = i
        policy_values += [tree.evaluate_policy(i)]

        qvals          = qval_tree[0][0]
        root_values   += [np.dot(tree._policy(qvals), qvals)]

    v_full = tree.full_updates()

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
        plt.ylim(0, v_full+0.1)
        plt.legend(prop={'size':13})

    plt.savefig(os.path.join(data_folder, 'root_values.png'))
    plt.savefig(os.path.join(data_folder, 'root_values.svg'), transparent=True)
    plt.close()

def plot_multiple(data_folder, M, P, R, nreps, R_true, horizons, xis, betas):

    fig, axes = plt.subplots(6, 2, figsize=(9, 18), dpi=100, constrained_layout=True, gridspec_kw={'height_ratios':[2, 2, 1, 2, 2, 1]})
    # plt.suptitle('alpha0 = %u, beta0 = %u, alpha1 = %u, beta1 = %u'%(alpha_0, beta_0, alpha_1, beta_1), fontsize=14)

    for hidx, h in enumerate(horizons):

        if (hidx == 0) or (hidx == 1): 
            axv = axes[0, hidx%2]
            axp = axes[1, hidx%2]
            axr = axes[2, hidx%2]
        else:
            axv = axes[3, hidx%2]
            axp = axes[4, hidx%2]
            axr = axes[5, hidx%2]

        for bidx, beta in enumerate(betas): 
            
            axv.plot(R[hidx, bidx, ::-1], label='Beta %.1f'%beta)
            axv.scatter(range(len(xis)), R[hidx, bidx, ::-1])

            axp.plot(P[hidx, bidx, ::-1], label='Beta %.1f'%beta)
            axp.scatter(range(len(xis)), P[hidx, bidx, ::-1])
            
            axr.plot(nreps[hidx, bidx, ::-1], label='Beta %.1f'%beta)
            axr.scatter(range(len(xis)), nreps[hidx, bidx, ::-1])

            if bidx == (len(betas) - 1):

                print(hidx, R_true)
                axv.axhline(R_true[hidx], linestyle='--', color='k', alpha=0.7, label='Optimal value')
                axp.axhline(R_true[hidx], linestyle='--', color='k', alpha=0.7, label='Optimal value')
            
                axv.legend(prop={'size': 13})
                # axp.legend(prop={'size': 13})
                # axr.legend(prop={'size': 13})

                axv.set_ylabel('Root value', fontsize=17)
                axv.set_ylim(0, np.max(R_true)+0.1)
                axv.set_title('Horizon %u'%(h-1), fontsize=18)
                axv.tick_params(axis='y', labelsize=13)

                axp.set_ylabel('Policy value', fontsize=17)
                axp.set_ylim(0, np.max(R_true)+0.1)
                axp.tick_params(axis='y', labelsize=13)

                axr.set_ylabel('Number of updates', fontsize=17)
                axr.tick_params(axis='y', labelsize=13)
                axr.set_ylim(0, np.nanmax(nreps)+6)

                axr.set_xlabel(r'$\xi$', fontsize=17)
                axr.set_xticks(range(R.shape[2]), ['%.4f'%i for i in xis[::-1]], rotation=60, fontsize=13)

                axv.set_xticks([])
                axp.set_xticks([])

    file_name = 'alpha0%u_beta0%u_alpha1%u_beta1%u_complete'%(M[0, 0], M[0, 1], M[1, 0], M[1, 1])
    np.save(os.path.join(data_folder, file_name + '.npy'), R)
    plt.savefig(os.path.join(data_folder, file_name + '.svg'), transparent=True)
    plt.savefig(os.path.join(data_folder, file_name + '.png'))
    plt.close()