from logging import root
import numpy as np
from belief_tree import Tree
from analysis import plot_root_values, plot_multiple
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
    'xi': 0.001,
    'beta': 4,
    'policy_type': 'softmax',
    'sequences': True,
    'max_seq_len': None,
    'horizon': 5
}

# save path
root_folder = '/home/georgy/Documents/Dayan_lab/PhD/bandits/bandit/data/new/'
save_path   = os.path.join(root_folder, '1', 'seqs')

# --- Main function for replay ---
def main_single_replay(save_path):
    tree = Tree(**p)

    qval_history, need_history, replay_history = tree.replay_updates()
    print('Number of replays: %u'%(len(replay_history)-1))
    print('Policy value: %.2f'%tree.evaluate_policy(tree.qval_tree))

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

    with open(os.path.join(save_path, 'tree.pkl'), 'wb') as f:
        pickle.dump(tree, f, pickle.HIGHEST_PROTOCOL)

    # save params
    with open(os.path.join(save_path, 'params.txt'), 'w') as f:
        f.write('Horizon:       %u\n'%tree.horizon)
        f.write('Prior belief: (alpha_0: %u, beta_0: %u, alpha_1: %u, beta_1: %u)\n'%(alpha_0, beta_0, alpha_1, beta_1))
        f.write('gamma:         %.2f\n'%tree.gamma)
        f.write('xi:            %.4f\n'%tree.xi)
        f.write('beta:          %.2f\n'%tree.beta)

    plot_root_values(save_path)

    return None

def main_multiple(save_path):
    
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    else: pass
    os.makedirs(save_path)
    os.mkdir(os.path.join(save_path, 'replay_data'))
    os.mkdir(os.path.join(save_path, 'data'))

    # vary these parameters
    xis       = np.logspace(np.log2(0.0001), np.log2(1.0), 11, base=2)
    horizons  = [2, 3, 4, 5]
    betas     = [1, 2, 4, 8]

    # store results here
    P      = np.zeros((len(horizons), len(betas), len(xis)))
    R      = np.zeros((len(horizons), len(betas), len(xis)))
    nreps  = np.zeros((len(horizons), len(betas), len(xis)), dtype=int)
    R_true = np.zeros(len(horizons))

    for hidx, horizon in enumerate(horizons):
                
        for bidx, beta in enumerate(betas):

            for xidx, xi in enumerate(xis):

                # initialise the agent
                p['beta']    = beta
                p['xi']      = xi
                p['horizon'] = horizon

                tree = Tree(**p)

                # do replay
                _, _, replays = tree.replay_updates()
                qvals         = tree.qval_tree[0][0].copy()
                v_replay      = np.dot(tree._policy(qvals), qvals)
                eval_pol      = tree.evaluate_policy(tree.qval_tree)

                P[hidx, bidx, xidx]     = eval_pol
                R[hidx, bidx, xidx]     = v_replay
                nreps[hidx, bidx, xidx] = len(replays)-1

                np.save(os.path.join(save_path, 'replay_data', 'replays_%u_%u_%u.npy'%(hidx, bidx, xidx)), replays)

                # do full bayesian updates
                if (bidx == 0) and (xidx == 0):
                    v_full       = tree.full_updates()
                    R_true[hidx] = v_full

        print('Horizon %u'%horizon)

    np.save(os.path.join(save_path, 'data', 'eval_pol.npy'), P)
    np.save(os.path.join(save_path, 'data', 'root_pol.npy'), R)
    np.save(os.path.join(save_path, 'data', 'full_upd.npy'), R_true)
    np.save(os.path.join(save_path, 'data', 'nreps.npy'), nreps)

    plot_multiple(save_path, p['root_belief'], P, R, nreps, R_true, horizons, xis, betas)

if __name__ == '__main__':
    main_multiple(save_path)