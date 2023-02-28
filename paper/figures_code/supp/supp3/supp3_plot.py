import numpy as np
import sys, os
sys.path.append('/home/georgy/Documents/Dayan_lab/PhD/bandits/paper/code/bandit')
import matplotlib.pyplot as plt

# --- Specify parameters ---

# save path
path = '/home/georgy/Documents/Dayan_lab/PhD/bandits/paper/figures/supp/supp3/'

# --- Main function for replay ---
def main():
    
    fig = plt.figure(figsize=(8, 5), dpi=100, constrained_layout=True)

    R_true = np.load(os.path.join(path, 'data', 'noseq_Rtrue.npy'))
    P      = np.load(os.path.join(path, 'data', 'noseq_P.npy'))
    R      = np.load(os.path.join(path, 'data', 'noseq_R.npy'))

    betas   = [1, 2, 4, 'greedy']
    xs      = [0.4, 0.8, 1.2, 1.6]
    colours = ['b', 'orange', 'g', 'r']

    for i in range(3):
        plt.subplot(2, 3, i+1)
        for bidx, beta in enumerate(betas):
            if bidx == 0:
                plt.axhline(R_true[i], c='k', linestyle='--', label='Optimal value')
                plt.title('Horizon %u'%(i+2), fontsize=14)
            plt.bar(xs[bidx], np.mean(P[:, i, bidx]), width=0.3, facecolor=colours[bidx], alpha=0.6)
            if i == 0:
                plt.scatter([xs[bidx]]*P.shape[0], P[:, i, bidx], label=r'$\beta=%s$'%beta, c=colours[bidx], alpha=1)
                plt.legend(prop={'size' : 6})
            else:
                plt.scatter([xs[bidx]]*P.shape[0], P[:, i, bidx], c=colours[bidx], alpha=1)
            plt.ylim(0, np.max(R_true)+0.6)
            plt.xlim(0.2, 1.8)
            plt.xticks([])
        if i == 0:
            plt.ylabel('Evaluated policy', fontsize=12)
    
    for i in range(3, 6):
        plt.subplot(2, 3, i+1)
        for bidx, beta in enumerate(betas):
            plt.bar(xs[bidx], np.mean(R[:, i%3, bidx]), width=0.3, facecolor=colours[bidx], alpha=0.6)
            plt.scatter([xs[bidx]]*R.shape[0], R[:, i%3, bidx], c=colours[bidx], alpha=1)
            plt.axhline(R_true[i%3], c='k', linestyle='--')
            plt.ylim(0, np.max(R_true)+0.6)
            plt.xlim(0.2, 1.8)
            plt.xticks([])
        if i == 3:
            plt.ylabel('Root value', fontsize=12)

    plt.savefig(os.path.join(path, 'supp3.png'))
    plt.savefig(os.path.join(path, 'supp3.svg'), transparent=True)
    plt.close()

    return None

if __name__ == '__main__':
    main()
