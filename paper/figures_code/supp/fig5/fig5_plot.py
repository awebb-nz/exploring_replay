import numpy as np
import matplotlib.pyplot as plt
import sys, os, pickle, shutil
sys.path.append('/home/georgy/Documents/Dayan_lab/PhD/bandits/maze/code/multiple_barriers')
from utils import plot_maze

load_path = '/home/georgy/Documents/Dayan_lab/PhD/bandits/paper/figures/fig4'
save_path = '/home/georgy/Documents/Dayan_lab/PhD/bandits/paper/figures/fig5'

def main():

    # if os.path.isdir(save_path):
    #     shutil.rmtree(save_path)
    #     os.makedirs(save_path)
    # else:
    #     os.makedirs(save_path)

    with open(os.path.join(load_path, 'ag.pkl'), 'rb') as f:
        agent = pickle.load(f)

    fig = plt.figure(figsize=(11, 10), constrained_layout=True, dpi=100)

    ax1 = fig.add_subplot(221)
    plot_maze(ax1, np.load(os.path.join(load_path, 'q_explore_online.npy')), agent, colorbar=True, colormap='Purples', move=[14])
    ax1.set_title(r'$Q^{MF}$ after online discovery', fontsize=14)
    ax1.text(-0.1, 1.1, 'A', transform=ax1.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

    ax2 = fig.add_subplot(222)
    plot_maze(ax2, np.load(os.path.join(load_path, 'q_explore_online_replay.npy')), agent, colorbar=True, colormap='Purples', move=[14])
    ax2.set_title(r'$Q^{MF}$ updated by replay', fontsize=14)
    ax2.text(-0.1, 1.1, 'B', transform=ax2.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

    ax3 = fig.add_subplot(223)
    q_explore_replay_diff = np.load(os.path.join(load_path, 'q_explore_online_replay_diff.npy'))
    q_explore_replay_diff[q_explore_replay_diff == 0.] = np.nan
    plot_maze(ax3, q_explore_replay_diff, agent, colorbar=True, colormap='Purples')
    ax3.set_title(r'Change in $Q^{MF}$ due to replay', fontsize=14)
    ax3.text(-0.1, 1.1, 'C', transform=ax3.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

    ax4  = fig.add_subplot(224)
    gain = np.load(os.path.join(load_path, 'gain_history.npy'), allow_pickle=True)[-1]
    gain[gain <= agent.xi] = np.nan
    gain = gain/np.nanmax(gain[:])
    plot_maze(ax4, gain, agent, colorbar=True, colormap='Greens')
    ax4.set_title('Normalised Gain for the second replay', fontsize=14)
    ax4.text(-0.1, 1.1, 'D', transform=ax4.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

    plt.savefig(os.path.join(save_path, 'fig5.png'))
    plt.savefig(os.path.join(save_path, 'fig5.svg'), transparent=True)
    plt.close()

    return None

if __name__ == '__main__':
    main()