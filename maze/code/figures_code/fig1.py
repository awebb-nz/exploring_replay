import numpy as np
import matplotlib.pyplot as plt
import sys, os, pickle, shutil
sys.path.append('/home/georgy/Documents/Dayan_lab/PhD/bandits/maze/code/multiple_barriers')
from utils import plot_maze, plot_need

data_path = '/home/georgy/Documents/Dayan_lab/PhD/bandits/maze/data/replay/local/tolman123/no_uncertainty/'
save_path = '/home/georgy/Documents/Dayan_lab/PhD/bandits/maze/data/paper_figures/fig1'

def main():
        
    with open(os.path.join(data_path, '0', 'ag.pkl'), "rb") as f:
        agent = pickle.load(f)

    num_seeds = len([i for i in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, i))])

    S  = [np.zeros((num_seeds, agent.num_states)), np.zeros((num_seeds, agent.num_states))]
    G  = [np.zeros((num_seeds, 6000, agent.num_states, agent.num_actions)), np.zeros((num_seeds, 6000, agent.num_states, agent.num_actions))]
    N  = [np.zeros((num_seeds, 6000, agent.num_states)), np.zeros((num_seeds, 6000, agent.num_states))]

    for idx, bounds in enumerate([[0, 3000], [3000, 6000]]):

        for seed in range(num_seeds):

            with open(os.path.join(data_path, str(seed), 'ag.pkl'), "rb") as f:
                agent = pickle.load(f)

            for file in range(bounds[0], bounds[1]):
                Gt           = {s:{a:[] for a in range(agent.num_actions)} for s in range(agent.num_states)}
                Nt           = {s:[] for s in range(agent.num_states)}
                
                data         = np.load(os.path.join(data_path, str(seed), 'Q_%u.npz'%file), allow_pickle=True)
                replays      = data['replays'][1:]
                gain_history = data['gain_history'][1:]
                need_history = data['need_history'][1:]

                if len(replays) > 0:
                    for ridx in range(len(replays)):

                        for sr in range(agent.num_states):
                            for ar in range(agent.num_actions):
                                Gt[sr][ar] += [gain_history[ridx][sr, ar]]
                                Nt[sr]     += [need_history[ridx][sr]]

                    for st in range(agent.num_states):
                        N[idx][seed, file, st] = np.max(Nt[st])
                        for at in range(agent.num_actions):
                            G[idx][seed, file, st, at] = np.max(Gt[st][at])

                move       = data['move']
                s          = int(move[3])
                S[idx][seed, s]   += 1

        G[idx]  = np.mean(G[idx], axis=(0, 1))
        N[idx]  = np.mean(N[idx], axis=(0, 1))
        S[idx]  = np.mean(S[idx], axis=0)

    fig = plt.figure(figsize=(19, 12), constrained_layout=True, dpi=100)

    agent.barriers = [1, 1, 0]
    ax1  = fig.add_subplot(231)
    plot_need(ax1, S[0], agent)
    ax1.set_title('State occupancy', fontsize=16)
    ax1.set_ylabel('Moves 1-3000', fontsize=16)
    ax1.text(-0.1, 1.1, 'A', transform=ax1.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

    ax2  = fig.add_subplot(232)
    plot_maze(ax2, G[0]/np.max(G[0]), agent)
    ax2.set_title('Normalised average max Gain', fontsize=16)
    ax2.text(-0.1, 1.1, 'B', transform=ax2.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

    ax3  = fig.add_subplot(233)
    plot_need(ax3, N[0], agent)
    ax3.set_title('Normalised average max Need', fontsize=16)
    ax3.text(-0.1, 1.1, 'C', transform=ax3.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

    agent.barriers = [0, 1, 0]
    ax4  = fig.add_subplot(234)
    plot_need(ax4, S[1], agent)
    ax4.set_ylabel('Moves 3000-6000', fontsize=16)
    ax4.text(-0.1, 1.1, 'D', transform=ax4.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

    ax5  = fig.add_subplot(235)
    plot_maze(ax5, G[1]/np.max(G[1]), agent)
    ax5.text(-0.1, 1.1, 'E', transform=ax5.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

    ax6  = fig.add_subplot(236)
    plot_need(ax6, N[1], agent)
    ax6.text(-0.1, 1.1, 'F', transform=ax6.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

    # ax6  = fig.add_subplot(236)
    # x    = np.arange(1000)
    # y    = np.cumsum(R[1][2000:])
    # ax6.plot(x, y, color='b', label='Agent')
    # ax6.fill_between(x, (y-ci[1][2000:]), (y+ci[1][2000:]), color='b', alpha=.4)
    # ax6.plot(np.cumsum(Rm[1][2000:]), color='k', label='Maximal')
    # ax6.set_xticks(range(0, 1001, 200), range(5000, 6001, 200), rotation=40)
    # ax6.set_xlabel('Move', fontsize=16)
    # ax6.set_ylabel('Cumulative reward', fontsize=16)
    # ax6.legend()
    # ax6.text(-0.1, 1.1, 'F', transform=ax6.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

    if os.path.isdir(save_path):
        shutil.rmtree(save_path)
        os.makedirs(save_path)
    else: pass

    plt.savefig(os.path.join(save_path, 'fig1.png'))
    plt.close()

    return None

if __name__ == '__main__':
    main()