import numpy as np
from agent_bamcp import Bamcp
import os

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
save_path  = '/home/georgy/Documents/Dayan_lab/PhD/bandits/maze/data/bamcp'
save_data  = os.path.join(save_path, 'moves')
save_plots = os.path.join(save_path, 'plots')

# --- Specify agent parameters ---
p = {
    'config':                   config,
    'num_actions':              4,
    'goal_coords':              goal_coords,
    'start_coords':             start_coords,
    'blocked_state_actions':    blocked_state_actions,
    'uncertain_states_actions': uncertain_states_actions,
    'gamma':                    0.90,
    'alpha':                    0.80,
    'num_sims':                 5000, 
    'eps':                      1e-7,
    'c':                        5,
    'M':                        np.array([1, 10]),
    'num_steps':                60
}

# --- Main function ---
def main():
    np.random.seed(0)

    # -------------------
    # --- BAMCP AGENT ---
    # -------------------

    agent = Bamcp(**p)
    # agent.run_simulation(save_path=save_path)
    agent.search(17)
    print(agent.tree[0][0])
    
    return None

if __name__ == '__main__':
    main()