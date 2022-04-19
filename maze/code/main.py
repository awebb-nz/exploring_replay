import numpy as np
from agent_bamcp import Bamcp
from agent_replay import Agent
from agent_need import BamcpNeed
from utils import plot_simulation
import matplotlib.pyplot as plt
import os
import pickle

# --- Specify the environment --- #
#                                 #
#        0  0  0  0  0  g         #
#        0  0  0  0  0  0         #
#           ----------- X         #
#        0  0  0  0  0  0         #
#        0  0  s  0  0  0         #
#                                 #
# # # # # # # # # # # # # # # # # # 

config       = np.zeros((4, 6))
num_states   = config.shape[0]*config.shape[1]
goal_coords  = [0, 5]
start_coords = [3, 2]
rew_value    = 1

config[goal_coords[0], goal_coords[1]] = rew_value

blocked_state_actions = [
                        [7,  1], [8,  1], [9,  1], [10, 1], [11, 1],
                        [13, 0], [14, 0], [15, 0], [16, 0], [17, 0], 
                        ]

uncertain_states_actions = [17, 0]

# --- Specify simulation parameters ---
#
num_steps  = 120
save_path  = '/home/georgy/Documents/Dayan_lab/PhD/bandits/maze/data/need'
save_data  = os.path.join(save_path, 'moves')
save_plots = os.path.join(save_path, 'plots')

# --- Specify agent parameters ---
gamma     = 0.85
alpha     = 1.0
alpha_r   = 1.0
beta      = 5
horizon   = 4 # minus 1
xi        = 1e-3
num_sdims = 1000

# prior belief about the barrier
M       = np.ones(2)

# --- Main function ---5
def main():
    np.random.seed(0)
    # initialise the agent
    agent = Agent(config, start_coords, goal_coords, blocked_state_actions, uncertain_states_actions, alpha, alpha_r, gamma, horizon, xi, policy_temp=beta)
    # run the simulation
    agent.run_simulation(num_steps=num_steps, start_replay=40, reset_prior=True, save_path=save_data)
    # save the agent
    with open(os.path.join(save_path, 'agent.pkl'), 'wb') as ag:
        pickle.dump(agent, ag, pickle.HIGHEST_PROTOCOL)
    # plot moves & replays
    plot_simulation(agent, save_data, save_plots)

    # Q = np.zeros((num_states, 4))
    # for s in np.delete(range(num_states), 5):
    #     agent = Bamcp(config, start_coords, goal_coords, blocked_state_actions, uncertain_states_actions, gamma)
    #     agent.search(s)
    #     print(s)
    #     tree    = agent.tree.copy()
    #     Q[s, :] = tree[0][0][1]
    #     # print(tree[0][0][2])
    #     # last_d = max(tree.keys())

    
    # np.save('/home/georgy/Documents/Dayan_lab/PhD/bandits/maze/data/bamcp/beta_1_1_round.npy', Q)
    # fig = plt.figure(figsize=(12, 6))
    # ax = plt.subplot(111)
    # plot_maze_values(Q, [], ax, config.shape[0], config.shape[1], 5, blocked_state_actions)
    # plt.savefig('/home/georgy/Documents/Dayan_lab/PhD/bandits/maze/data/bamcp/beta_1_1_round.png')
    
    # Q = np.load('/home/georgy/Documents/Dayan_lab/PhD/bandits/maze/data/moves/Q_3000.npz')['Q_history'][0]
    # agent = BamcpNeed(config, start_coords, goal_coords, blocked_state_actions, uncertain_states_actions, gamma, Q, M, policy_temp=4)
    # agent.search(20)
    # SR = agent.SR
    # np.save('/home/georgy/Documents/Dayan_lab/PhD/bandits/maze/data/bamcp/need/need.npy', SR)
    
    return None

if __name__ == '__main__':
    main()