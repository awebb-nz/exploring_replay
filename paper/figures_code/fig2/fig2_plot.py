import numpy as np
import matplotlib.pyplot as plt
import sys, os, pickle
sys.path.append('/home/georgy/Documents/Dayan_lab/PhD/bandits/maze/code/multiple_barriers')
from utils import plot_maze

load_path = '/home/georgy/Documents/Dayan_lab/PhD/bandits/paper/figures/fig2'

def main():

    with open(os.path.join(load_path, 'ag.pkl'), 'rb') as f:
        agent = pickle.load(f)

    fig = plt.figure(figsize=(14, 8), constrained_layout=True, dpi=100)

    ax1 = fig.add_subplot(2,3,1)
    plot_maze(ax1, np.load(os.path.join(load_path, 'q_mb.npy')), agent, colorbar=True, colormap='Purples', move=[38])
    ax1.set_title(r'Initial behavioural policy', fontsize=16)
    ax1.text(-0.1, 1.1, 'A', transform=ax1.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

    ax2 = fig.add_subplot(2,3,2)
    q_explore_replay_diff = np.load(os.path.join(load_path, 'q_explore_replay_diff.npy'))
    q_explore_replay_diff[q_explore_replay_diff <= agent.xi] = np.nan
    plot_maze(ax2, q_explore_replay_diff, agent, colorbar=True, colormap='Purples')
    ax2.set_title(r'Exploratory replay', fontsize=16)
    ax2.text(-0.1, 1.1, 'B', transform=ax2.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

    ax3 = fig.add_subplot(2,3,3)
    plot_maze(ax3, np.load(os.path.join(load_path, 'q_explore_replay.npy')), agent, colorbar=True, colormap='Purples', move=[38])
    ax3.set_title(r'Updated exploratory policy', fontsize=16)
    ax3.text(-0.1, 1.1, 'C', transform=ax3.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

    # ax4 = fig.add_subplot(4,3,5)

    # M_range = [[0, 1], [1, 7], [1, 5], [1, 3], [1, 1], [3, 1], [5, 1], [7, 1], [1, 0]]
    # betas   = [1, 5, 10, 20]
    # x       = np.arange(len(M_range))

    # qs      = np.load(os.path.join(load_path, 'qs_explore.npy'))
    
    # for beta in betas:

    #     agent.beta = beta
    #     probs = np.zeros(len(M_range))
    #     for i in range(0, len(qs), 8):
    #         probs[i//8] = agent._policy(qs[i])[0]*agent._policy(qs[i+1])[0]*agent._policy(qs[i+2])[3]*agent._policy(qs[i+3])[3]*agent._policy(qs[i+4])[0]*agent._policy(qs[i+5])[0]*agent._policy(qs[i+6])[2]*agent._policy(qs[i+7])[2]
    #     ax4.plot(x, probs, label=r'$\beta=%u$'%beta)

    # probs = np.load(os.path.join(load_path, 'probas_greedy.npy'))
    # for i in range(7, len(qs), 8):
    #     probs[i//8] = (qs[i][2] == np.nanmax(qs[i]))
    # ax4.plot(x, probs, label='greedy')

    # ax4.set_ylim(-0.05, 1.05)
    # ax4.set_xlim(0, 1)

    # ax4.set_xticks(x, np.linspace(0, 1, len(x)), fontsize=12, rotation=30)
    # ax4.set_xlabel('Belief', fontsize=12)
    # ax4.set_title('Exploration quality of the new policy', fontsize=12)
    # ax4.set_ylabel('Exploration probability', fontsize=12)
    # ax4.legend(prop={'size':12})
    # ax4.text(-0.1, 1.1, 'D', transform=ax4.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

    ax5 = fig.add_subplot(2,3,4)
    plot_maze(ax5, np.load(os.path.join(load_path, 'q_explore_online.npy')), agent, colorbar=True, colormap='Purples', move=[14])
    ax5.set_title(r'Online discovery', fontsize=16)
    ax5.text(-0.1, 1.1, 'D', transform=ax5.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

    ax6 = fig.add_subplot(2,3,5)
    q_explore_replay_diff = np.load(os.path.join(load_path, 'q_explore_online_replay_diff.npy'))
    q_explore_replay_diff[q_explore_replay_diff == 0.] = np.nan
    plot_maze(ax6, q_explore_replay_diff, agent, colorbar=True, colormap='Purples')
    ax6.set_title(r'Replay', fontsize=16)
    ax6.text(-0.1, 1.1, 'E', transform=ax6.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

    ax7 = fig.add_subplot(2,3,6)
    plot_maze(ax7, np.load(os.path.join(load_path, 'q_explore_online_replay.npy')), agent, colorbar=True, colormap='Purples', move=[14])
    ax7.set_title(r'Updated policy', fontsize=16)
    ax7.text(-0.1, 1.1, 'F', transform=ax7.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

    # ax8  = fig.add_subplot(4,3,11)
    # gain = np.load(os.path.join(load_path, 'gain_history.npy'), allow_pickle=True)[-1]
    # gain[gain <= agent.xi] = np.nan
    # gain = gain/np.nanmax(gain[:])
    # plot_maze(ax8, gain, agent, colorbar=True, colormap='Greens')
    # ax8.set_title('Normalised Gain for the second replay', fontsize=12)
    # ax8.text(-0.1, 1.1, 'H', transform=ax8.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

    plt.savefig(os.path.join(load_path, 'fig2.png'))
    plt.savefig(os.path.join(load_path, 'fig2.svg'), transparent=True)

    return None

if __name__ == '__main__':
    main()