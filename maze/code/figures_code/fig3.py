import numpy as np
import matplotlib.pyplot as plt
import sys, os, shutil
sys.path.append('/home/georgy/Documents/Dayan_lab/PhD/bandits/maze/code/multiple_barriers')
from agent_replay import AgentPOMDP
from utils import plot_maze, load_env

np.random.seed(2)

env            = 'tolman123'
env_file_path  = '/home/georgy/Documents/Dayan_lab/PhD/bandits/maze/code/mazes/' + env + '.txt'
env_config     = load_env(env_file_path)

# --- Specify agent parameters ---
pag_config = {
    'alpha'          : 1,
    'beta'           : 10, 
    'need_beta'      : 10,
    'gain_beta'      : 60,          
    'gamma'          : 0.9,
    'policy_type'    : 'softmax'
}

ag_config = {
    'alpha_r'        : 1,         # offline learning rate
    'horizon'        : 12,        # planning horizon (minus 1)
    'xi'             : 0.03,     # EVB replay threshold
    'num_sims'       : 2000,      # number of MC simulations for need
    'sequences'      : True,
    'max_seq_len'    : 4,        
    'env_name'       : env,       # gridworld name
    'barriers'       : [1, 0, 0]
}

save_path = '/home/georgy/Documents/Dayan_lab/PhD/bandits/maze/data/paper_figures/fig3'

def main():

    if os.path.isdir(save_path):
        shutil.rmtree(save_path)
        os.makedirs(save_path)
    else: pass

    agent = AgentPOMDP(*[pag_config, ag_config, env_config])

    Q_MB  = agent._solve_mb(1e-5, barriers=[1, 0, 0])

    fig = plt.figure(figsize=(11, 20), constrained_layout=True, dpi=100)

    ax1 = fig.add_subplot(421)
    plot_maze(ax1, Q_MB, agent, colorbar=True)
    ax1.set_title('Q-values', fontsize=14)
    ax1.set_ylabel('Initial Q^{MF}-values', fontsize=14)
    ax1.text(-0.1, 1.1, 'A', transform=ax1.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

    agent.state = 38          # start state
    agent.M     = np.array([[1, 0], [1, 0], [1, 0]])
    agent.Q     = Q_MB.copy() # set MF Q values
    Q_history, gain_history, need_history = agent._replay()

    ax2 = fig.add_subplot(423)
    plot_maze(ax2, agent.Q, agent, colorbar=False)
    ax2.set_ylabel('Exploratory replay', fontsize=14)
    ax2.text(-0.1, 1.1, 'B', transform=ax2.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

    ax3 = fig.add_subplot(424)
    plot_maze(ax3, agent.Q - Q_MB, agent, colorbar=True)
    ax3.set_title('Change in Q-values', fontsize=14)

    Q              = agent.Q.copy()
    Q_before       = Q.copy()
    Q_after        = Q.copy()
    Q_after[14, 0] = 0.0
    agent.Q        = Q_after.copy()

    ax4 = fig.add_subplot(425)
    plot_maze(ax4, agent.Q, agent, colorbar=False)
    ax4.set_ylabel('Online discovery', fontsize=14)
    ax4.text(-0.1, 1.1, 'C', transform=ax4.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

    ax5 = fig.add_subplot(426)
    plot_maze(ax5, agent.Q - Q_before, agent, colorbar=True)

    Q_before    = agent.Q.copy()

    agent.state = 14
    agent.M     = np.array([[0, 1], [1, 0], [1, 0]])
    Q_history, gain_history, need_history = agent._replay()

    ax6 = fig.add_subplot(427)
    plot_maze(ax6, agent.Q, agent, colorbar=False)
    ax6.set_ylabel('Subsequent replay', fontsize=14)
    ax6.text(-0.1, 1.1, 'D', transform=ax6.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

    ax7 = fig.add_subplot(428)
    plot_maze(ax7, agent.Q - Q_before, agent, colorbar=True)
    
    plt.savefig(os.path.join(save_path, 'fig3.png'))
    plt.close()

    return None

if __name__ == '__main__':
    main()