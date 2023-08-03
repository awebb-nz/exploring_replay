import numpy as np
import sys, os, shutil, pickle
from copy import deepcopy
sys.path.append('/home/georgy/Documents/Dayan_lab/PhD/bandits/paper/code/bandit')
from belief_tree import Tree
import matplotlib.pyplot as plt

# --- Specify parameters ---

# prior belief at the root
alpha_0, beta_0 = 1, 1

M = {
    0: np.array([alpha_0, beta_0]), 
    1: 0.51
    }

# other parameters
p = {
    'arms':           ['unknown', 'known'],
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

xis       = np.logspace(np.log2(0.001), np.log2(1), 11, base=2)
betas     = [1, 2, 4, 'greedy']
horizons  = [3, 4, 5]

# save path
save_path = '/home/georgy/Documents/Dayan_lab/PhD/bandits/paper/figures/fig2_new/data/d_f'

def md_value():

    out = np.zeros(len(horizons))
    x   = deepcopy(p)
    for idx, horizon in enumerate(horizons[::-1]):

        x['horizon'] = horizon
        x['root_belief'] = {0: 0.50, 
                            1: 0.51}
        x['arms'] = ['known', 'known']
        # initialise the agent
        tree      = Tree(**x)
        
        # do full bayesian updates
        qval_tree = tree.full_updates()
        qvals     = qval_tree[0][0]
        v_full    = np.max(qvals)

        out[idx]  = v_full

    return out

# --- Main function for replay ---
def main(save_path):
    
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    else: pass
    os.makedirs(save_path)

    md_values = md_value()

    for hidx, horizon in enumerate(horizons[::-1]):
        # store results here
        P      = np.zeros((len(betas), len(xis)))
        R      = np.zeros((len(betas), len(xis)))
        nreps  = np.zeros((len(betas), len(xis)), dtype=int)

        p['horizon'] = horizon
        # initialise the agent
        tree      = Tree(**p)
        
        # do full bayesian updates
        qval_tree = tree.full_updates()
        qvals     = qval_tree[0][0]
        v_full    = np.max(qvals)
        
        if horizon == 5:
            R_true    = v_full

        for bidx, beta in enumerate(betas):

            for xidx, xi in enumerate(xis):
                
                # initialise the agent
                p['beta'] = beta
                p['xi']   = xi
                tree      = Tree(**p)
                
                # do replay
                _, _, _, replays = tree.replay_updates()
                qvals            = tree.qval_tree[0][0]
                v_replay         = tree._value(qvals)

                eval_pol         = tree.evaluate_policy(tree.qval_tree)

                P[bidx, xidx]     = eval_pol
                R[bidx, xidx]     = v_replay
                nreps[bidx, xidx] = len(replays)-1

        fig, axes = plt.subplots(3, 1, figsize=(3, 5), dpi=100, constrained_layout=True, gridspec_kw={'height_ratios':[1, 1, 1]})

        axv = axes[0]
        axp = axes[1]
        axr = axes[2]

        for bidx, beta in enumerate(betas): 
            if beta == 'greedy':
                lab = 'greedy'
            else:
                lab = r'$\beta=$%s'%beta

            axv.plot(R[bidx, ::-1], label=lab)
            axv.scatter(range(len(xis)), R[bidx, ::-1])

            axp.plot(P[bidx, ::-1], label=lab)
            axp.scatter(range(len(xis)), P[bidx, ::-1])

            axr.plot(nreps[bidx, ::-1], label=lab)
            axr.scatter(range(len(xis)), nreps[bidx, ::-1])

            if bidx == (len(betas) - 1):

                print(v_full)
                axv.set_title('Horizon %u'%horizon, fontsize=13)
                axv.axhline(v_full, linestyle='--', color='k', alpha=0.7, label='BO value')
                axv.axhline(md_values[hidx], linestyle='--', color='r', alpha=0.7, label='CE value')

                axp.axhline(v_full, linestyle='--', color='k', alpha=0.7, label='BO value')
                axp.axhline(md_values[hidx], linestyle='--', color='r', alpha=0.7, label='CE value')

                axr.legend(prop={'size': 9})
                axv.legend(prop={'size': 9})
                axp.legend(prop={'size': 9})
                # axp.legend(prop={'size': 13})
                # axr.legend(prop={'size': 13})

                axv.set_ylabel('Root value', fontsize=12)
                axv.set_ylim(0, R_true+1)
                axv.tick_params(axis='y', labelsize=11)

                axp.set_ylabel('Policy value', fontsize=12)
                axp.set_ylim(R_true-1, R_true+0.5)
                axp.tick_params(axis='y', labelsize=11)

                if horizon == 5:
                    max_reps = np.max(nreps[:])

                axr.set_ylabel('Number of updates', fontsize=12)
                axr.tick_params(axis='y', labelsize=10)
                axr.set_ylim(0, max_reps+6)

                axr.set_xlabel(r'$\xi$', fontsize=12)
                axr.set_xticks(range(R.shape[1]), ['%.4f'%i for i in xis[::-1]], rotation=60, fontsize=10)

                axv.set_xticks([])
                axp.set_xticks([])
                
        file_name = 'arm1%u_alpha0%u_beta0%u_hor%u'%(M[1], alpha_0, beta_0, horizon)
        np.save(os.path.join(save_path, file_name + '.npy'), R)
        plt.savefig(os.path.join(save_path, file_name + '.svg'), transparent=True)
        plt.savefig(os.path.join(save_path, file_name + '.png'))

        plt.close()

    return None

if __name__ == '__main__':
    main(save_path)
