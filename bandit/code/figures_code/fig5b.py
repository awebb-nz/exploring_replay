import numpy as np
import sys, os, shutil, pickle
sys.path.append('/home/georgy/Documents/Dayan_lab/PhD/bandits/bandit/code')
from belief_tree import Tree
import matplotlib.pyplot as plt

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
    'rand_init':      False,
    'gamma':          0.9,
    'xi':             0.0001,
    'beta':           4,
    'policy_type':    'softmax',
    'sequences':      False,
    'max_seq_len':    None,
    'constrain_seqs': True,
    'horizon':        4
}

# save path
save_path = '/home/georgy/Documents/Dayan_lab/PhD/bandits/bandit/data/paper_figures/fig5'

# --- Main function for replay ---
def main():
    
    # vary these parameters
    # xis       = np.append(0, np.logspace(np.log2(0.001), np.log2(1.0), 10, base=2))
    xis       = np.logspace(np.log2(0.0001), np.log2(1.0), 11, base=2)
    betas     = [1.5, 2, 4]

    # store results here
    P      = np.zeros((len(betas), len(xis)))
    R      = np.zeros((len(betas), len(xis)))
    nreps  = np.zeros((len(betas), len(xis)), dtype=int)
    R_true = 0

    # initialise the agent
    tree   = Tree(**p)
    
    # do full bayesian updates
    qval_tree = tree.full_updates()
    qvals     = qval_tree[0][0]
    v_full    = np.max(qvals)
    R_true    = v_full

    for bidx, beta in enumerate(betas):

        for xidx, xi in enumerate(xis):
            
            # initialise the agent
            p['beta'] = beta
            p['xi']   = xi
            tree      = Tree(**p)
            
            # do replay
            _, _, replays = tree.replay_updates()
            qvals         = tree.qval_tree[0][0]
            v_replay      = tree._value(qvals)

            eval_pol      = tree.evaluate_policy(tree.qval_tree)

            P[bidx, xidx]     = eval_pol
            R[bidx, xidx]     = v_replay
            nreps[bidx, xidx] = len(replays)-1

    fig, axes = plt.subplots(3, 1, figsize=(5, 9), dpi=100, constrained_layout=True, gridspec_kw={'height_ratios':[1, 1, 1]})

    hidx = 2

    axv = axes[0]
    axp = axes[1]
    axr = axes[2]

    for bidx, beta in enumerate(betas): 
            
        axv.plot(R[bidx, ::-1], label='Beta %.1f'%beta)
        axv.scatter(range(len(xis)), R[bidx, ::-1])

        axp.plot(P[bidx, ::-1], label='Beta %.1f'%beta)
        axp.scatter(range(len(xis)), P[bidx, ::-1])

        axr.plot(nreps[bidx, ::-1], label='Beta %.1f'%beta)
        axr.scatter(range(len(xis)), nreps[bidx, ::-1])

        if bidx == (len(betas) - 1):

            print(R_true)
            axv.axhline(R_true, linestyle='--', color='k', alpha=0.7, label='Optimal value')
            axp.axhline(R_true, linestyle='--', color='k', alpha=0.7, label='Optimal value')

            axv.legend(prop={'size': 10})
            # axp.legend(prop={'size': 13})
            # axr.legend(prop={'size': 13})

            axv.set_ylabel('Root value', fontsize=17)
            axv.set_ylim(0, np.max(R_true)+0.1)
            axv.tick_params(axis='y', labelsize=13)

            axp.set_ylabel('Policy value', fontsize=17)
            axp.set_ylim(1.9, np.max(R_true)+0.1)
            axp.tick_params(axis='y', labelsize=13)

            axr.set_ylabel('Number of updates', fontsize=17)
            axr.tick_params(axis='y', labelsize=13)
            axr.set_ylim(0, np.nanmax(nreps)+6)

            axr.set_xlabel(r'$\xi$', fontsize=17)
            axr.set_xticks(range(R.shape[1]), ['%.4f'%i for i in xis[::-1]], rotation=60, fontsize=10)

            axv.set_xticks([])
            axp.set_xticks([])
            
    file_name = 'alpha0%u_beta0%u_alpha1%u_beta1%u_complete'%(alpha_0, beta_0, alpha_1, beta_1)
    np.save(os.path.join(save_path, file_name + '.npy'), R)
    plt.savefig(os.path.join(save_path, file_name + '.svg'), transparent=True)
    plt.savefig(os.path.join(save_path, file_name + '.png'))

    plt.close()

    return None

if __name__ == '__main__':
    main()
