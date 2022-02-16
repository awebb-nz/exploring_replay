import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import RegularPolygon
from matplotlib.collections import PatchCollection
import seaborn as sns
import os, glob, shutil
sns.set_style('white')

def plot_simulation(agent, data_path, save_path):
    
    if os.path.isdir(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)

    # --- simulation parameters ---
    num_x_states          = agent.num_x_states
    num_y_states          = agent.num_y_states
    goal_state            = agent.goal_state
    blocked_state_actions = agent.blocked_state_actions

    files = glob.glob(os.path.join(data_path, '*.npz'))
    files.sort(key=lambda s: int(s.split('_')[-1][:-4]))

    for file in files:
        data         = np.load(file, allow_pickle=True)
        Q_history    = data['Q_history']
        gain_history = data['gain_history']
        need_history = data['need_history']
        move         = data['move']

        fig = plt.figure(figsize=(27, 16), constrained_layout=True)
        ax  = plt.subplot(221)
        plot_maze_values(Q_history[0], move, ax, num_y_states, num_x_states, goal_state, blocked_state_actions)
        ax1 = plt.subplot(223)
        plot_replay_action([], ax1, num_y_states, num_x_states, goal_state, blocked_state_actions)
        ax2 = plt.subplot(222)
        plot_replay_action([], ax2, num_y_states, num_x_states, goal_state, blocked_state_actions)
        ax3 = plt.subplot(224)
        plot_replay_action([], ax3, num_y_states, num_x_states, goal_state, blocked_state_actions)
        plt.savefig(os.path.join(save_path, 'move_%s.png')%file.split('_')[-1][:-4])
        plt.close()

        for idx, Q_rep in enumerate(Q_history[1:]):
            if idx == 0:
                sr, ar = np.argwhere(np.nan_to_num(Q_rep, nan=0) != np.nan_to_num(Q_history[0], nan=0)).flatten()
            else:
                sr, ar = np.argwhere(np.nan_to_num(Q_history[idx], nan=0) != np.nan_to_num(Q_rep, nan=0)).flatten()
            fig = plt.figure(figsize=(27, 16), constrained_layout=True)
            ax  = plt.subplot(221)
            plot_maze_values(Q_rep, move, ax, num_y_states, num_x_states, goal_state, blocked_state_actions)
            ax1 = plt.subplot(223)
            plot_replay_action([sr, ar], ax1, num_y_states, num_x_states, goal_state, blocked_state_actions)
            ax2 = plt.subplot(222)
            plot_maze_values(gain_history[idx+1], [], ax2, num_y_states, num_x_states, goal_state, blocked_state_actions)
            ax3 = plt.subplot(224)
            plot_need_values(need_history[idx+1], ax3, num_y_states, num_x_states, goal_state, blocked_state_actions)
            plt.savefig(os.path.join(save_path, 'move_%s_%u.png')%(file.split('_')[-1][:-4], idx))
            plt.close()

    return None

def add_patches(s, a, q, num_y_states, num_x_states):
    
    num_states = num_y_states * num_x_states

    patches = []
    
    if q >= 0:
        col_tuple = (1, 0, 0, q)
    else:
        col_tuple = (0, 0, 1, -q)
        
    i, j = np.argwhere(np.arange(num_states).reshape(num_y_states, num_x_states) == s).flatten()
    
    # move up
    if a == 0:
        patches.append(RegularPolygon((0.5+j, num_y_states-0.18-i), 3, radius=0.1, lw=0.5, orientation=0, edgecolor='k', fill=True, facecolor=col_tuple))
    # move down
    elif a == 1:
        patches.append(RegularPolygon((0.5+j, num_y_states-0.82-i), 3, radius=0.1, lw=0.5, orientation=np.pi, edgecolor='k', fill=True, facecolor=col_tuple))
    # move left
    elif a == 2:
        patches.append(RegularPolygon((0.20+j, num_y_states-0.49-i), 3, radius=0.1, lw=0.5, orientation=np.pi/2, edgecolor='k', fill=True, facecolor=col_tuple))
    # move right
    else:
        patches.append(RegularPolygon((0.80+j, num_y_states-0.49-i), 3, radius=0.1, lw=0.5, orientation=-np.pi/2, edgecolor='k', fill=True, facecolor=col_tuple))
                    
    return patches

def plot_maze_values(Q, move, ax, num_y_states, num_x_states, goal_state, blocked_state_actions):
    
    num_states = num_y_states * num_x_states

    Q_plot = np.nanmax(Q, axis=1).reshape(num_y_states, num_x_states)[::-1, :]
    if np.all(Q_plot == 0):
        sns.heatmap(Q_plot, cmap=['white'], annot=False, fmt='.2f', cbar=True, ax=ax)
    else:
        Q_plot = Q_plot/np.nanmax(Q_plot)
        sns.heatmap(Q_plot, cmap='Greys', annot=True, fmt='.2f', cbar=True, ax=ax)
    
    # arrows for actions
    patches = []
    for st in np.delete(range(num_states), goal_state):
        for ac in range(4):
            if ~np.isnan(Q[st, ac]):
                patches += add_patches(st, ac, Q[st, ac]/np.nanmax(np.abs(Q)), num_y_states, num_x_states)
            if [st, ac] in blocked_state_actions:
                i, j = np.argwhere(np.arange(num_states).reshape(num_y_states, num_x_states) == st).flatten()
                if ac == 0:
                    ax.hlines((num_y_states-i), j, j+1, linewidth=6, color='b')
                elif ac == 3:
                    ax.vlines(j+1, (num_y_states-i)-1, (num_y_states-i), linewidth=6, color='b')

    collection = PatchCollection(patches, match_original=True)
    ax.add_collection(collection)
                
    # state grid
    for st_x in range(num_x_states):
        for st_y in range(num_y_states):
            ax.axhline(st_y, c='k', linewidth=0.6)
            ax.axvline(st_x, c='k', linewidth=0.6)

    # goal symbol
    goal_y, goal_x   = np.argwhere(np.arange(num_states).reshape(num_y_states, num_x_states) == goal_state).flatten()
    ax.scatter(goal_x+0.5, num_y_states - goal_y -0.5, s=600, c='orange', marker=r'$\clubsuit$', alpha=0.7)

    # agent location
    if len(move) > 0:
        agent_state      = move[-1]
        agent_y, agent_x = np.argwhere(np.arange(num_states).reshape(num_y_states, num_x_states) == agent_state).flatten()
        ax.scatter(agent_x+0.5, num_y_states - agent_y -0.5, s=600, c='green', alpha=0.7)

    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_xlim(0, num_x_states)
    ax.set_ylim(0, num_y_states)

    if len(move) > 0:
        ax.set_title('[' + ' '.join(map(str, [int(i) for i in move])) + ']', fontsize=20)
    else:
        ax.set_title('Gain', fontsize=20)
    return None

def plot_replay_action(move, ax, num_y_states, num_x_states, goal_state, blocked_state_actions):

    num_states = num_y_states * num_x_states

    Q_plot = np.zeros((num_states, 4))
    if np.all(Q_plot == 0):
        sns.heatmap(Q_plot, cmap=['white'], annot=False, fmt='.2f', cbar=True, ax=ax)
    else:
        sns.heatmap(Q_plot, cmap='Greys', annot=True, fmt='.2f', cbar=True, ax=ax)
    
    # arrows for actions
    patches = []
    for st in np.delete(range(num_states), goal_state):
        for ac in range(4):
            if len(move) > 0:
                if (st == move[0]) and (ac == move[1]):
                    patches += add_patches(st, ac, 1, num_y_states, num_x_states)
            else:
                pass
            if [st, ac] in blocked_state_actions:
                i, j = np.argwhere(np.arange(num_states).reshape(num_y_states, num_x_states) == st).flatten()
                if ac == 0:
                    ax.hlines((num_y_states-i), j, j+1, linewidth=6, color='b')
                elif ac == 3:
                    ax.vlines(j+1, (num_y_states-i)-1, (num_y_states-i), linewidth=6, color='b')

    collection = PatchCollection(patches, match_original=True)
    ax.add_collection(collection)
                
    # state grid
    for st_x in range(num_x_states):
        for st_y in range(num_y_states):
            ax.axhline(st_y, c='k', linewidth=0.6)
            ax.axvline(st_x, c='k', linewidth=0.6)

    goal_y, goal_x   = np.argwhere(np.arange(num_states).reshape(num_y_states, num_x_states) == goal_state).flatten()

    # goal symbol
    ax.scatter(goal_x+0.5, num_y_states - goal_y -0.5, s=600, c='orange', marker=r'$\clubsuit$', alpha=0.7)

    # agent location
    if len(move) > 0:
        agent_state      = move[0]
        agent_y, agent_x = np.argwhere(np.arange(num_states).reshape(num_y_states, num_x_states) == agent_state).flatten()
    
        # agent location
        ax.scatter(agent_x+0.5, num_y_states - agent_y -0.5, s=600, c='green', alpha=0.7)
        ax.set_title('[' + ' '.join(map(str, [int(i) for i in move])) + ']', fontsize=20)

    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_xlim(0, num_x_states)
    ax.set_ylim(0, num_y_states)

    return None

def plot_need_values(need, ax, num_y_states, num_x_states, goal_state, blocked_state_actions):
    
    num_states = num_y_states * num_x_states

    need_plot = need.reshape(num_y_states, num_x_states)[::-1, :]
    need_plot = need_plot/np.nanmax(need_plot)
    if np.all(need_plot == 0):
        sns.heatmap(need_plot, cmap=['white'], annot=False, fmt='.2f', cbar=True, ax=ax)
    else:
        sns.heatmap(need_plot, cmap='Blues', annot=True, fmt='.2f', cbar=True, ax=ax)
    
    # arrows for actions
    patches = []
    for st in np.delete(range(num_states), goal_state):
        for ac in range(4):
            if [st, ac] in blocked_state_actions:
                i, j = np.argwhere(np.arange(num_states).reshape(num_y_states, num_x_states) == st).flatten()
                if ac == 0:
                    ax.hlines((num_y_states-i), j, j+1, linewidth=6, color='b')
                elif ac == 3:
                    ax.vlines(j+1, (num_y_states-i)-1, (num_y_states-i), linewidth=6, color='b')

    collection = PatchCollection(patches, match_original=True)
    ax.add_collection(collection)
                
    # state grid
    for st_x in range(num_x_states):
        for st_y in range(num_y_states):
            ax.axhline(st_y, c='k', linewidth=0.6)
            ax.axvline(st_x, c='k', linewidth=0.6)

    # goal symbol
    goal_y, goal_x   = np.argwhere(np.arange(num_states).reshape(num_y_states, num_x_states) == goal_state).flatten()
    ax.scatter(goal_x+0.5, num_y_states - goal_y -0.5, s=600, c='orange', marker=r'$\clubsuit$', alpha=0.7)

    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_xlim(0, num_x_states)
    ax.set_ylim(0, num_y_states)

    ax.set_title('Need', fontsize=20)

    return None