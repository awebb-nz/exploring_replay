import numpy as np
import matplotlib.pyplot as plt
import sys, os, pickle
sys.path.append('/home/georgy/Documents/Dayan_lab/PhD/bandits/maze/code/multiple_barriers')
from utils import plot_maze

load_path = '/home/georgy/Documents/Dayan_lab/PhD/bandits/paper/figures/supp/supp8'

def main():

    with open(os.path.join(load_path, 'ag.pkl'), 'rb') as f:
        agent = pickle.load(f)

    fig = plt.figure(figsize=(13, 8), constrained_layout=True, dpi=100)

    ax1 = fig.add_subplot(2,3,1)
    plot_maze(ax1, np.load(os.path.join(load_path, 'q_mb.npy')), agent, colorbar=True, colormap='Purples', move=[38])
    ax1.set_title(r'Initial behavioural policy', fontsize=16)
    ax1.text(-0.1, 1.1, 'A', transform=ax1.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

    gain_history = np.load(os.path.join(load_path, 'gain_history.npy'), allow_pickle=True)

    ax2  = fig.add_subplot(2,3,2)
    Gain = agent.Q_nans 
    Gain[14, 0] = gain_history[1][0][8][0]
    plot_maze(ax2, Gain, agent, colorbar=True, colormap='Purples')
    ax2.set_title(r'Gain for open barrier', fontsize=16)
    ax2.text(-0.1, 1.1, 'B', transform=ax2.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

    ax3 = fig.add_subplot(2,3,5)
    Gain = agent.Q_nans 
    Gain[14, 0] = gain_history[1][0][8][1]
    plot_maze(ax3, Gain, agent, colorbar=True, colormap='Purples', move=[38])
    ax3.set_title(r'Gain for closed barrier', fontsize=16)
    # ax3.text(-0.1, 1.1, 'C', transform=ax3.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

    plt.savefig(os.path.join(load_path, 'supp8.png'))
    plt.savefig(os.path.join(load_path, 'supp8.svg'), transparent=True)

    return None

if __name__ == '__main__':
    main()