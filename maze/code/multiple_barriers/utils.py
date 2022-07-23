import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import RegularPolygon, Rectangle
from matplotlib.collections import PatchCollection
import seaborn as sns
import os, glob, shutil, ast
sns.set_style('white')

def load_env(env_file_path):
    with open(env_file_path, 'r') as f:
        env_config = {}
        for line in f:
            k, v = line.strip().split('=')
            env_config[k.strip()] = ast.literal_eval(v.strip())
    
    return env_config 

def plot_simulation(agent, data_path, save_path, move_start=None):
    
    if os.path.isdir(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)

    files = glob.glob(os.path.join(data_path, '*.npz'))
    files.sort(key=lambda s: int(s.split('_')[-1][:-4]))

    if move_start is not None:
        files = files[move_start:]

    for file in files:
        data         = np.load(file, allow_pickle=True)
        Q_history    = data['Q_history']

        if 'gain_history' in data.files:
            gain_history = data['gain_history']
            need_history = data['need_history']
            replay       = True
        else:
            replay       = False

        move = data['move']

        fig = plt.figure(figsize=(27, 16), constrained_layout=True)
        # plot Q values
        ax  = plt.subplot(221)
        if replay:
            plot_maze(ax, Q_history[0], agent, move)
        else:
            plot_maze(ax, Q_history, agent, move)
        # plot Replay
        ax1 = plt.subplot(223)
        # plot_replay(ax1, agent, move=None)
        # plot 
        ax2 = plt.subplot(222)
        # plot_maze(ax2, gain, agent, move=None)
        ax3 = plt.subplot(224)
        # plot_need(ax3, need, agent)

        plt.savefig(os.path.join(save_path, 'move_%s.png')%file.split('_')[-1][:-4])
        plt.close()

        if replay:
            for idx, Q_rep in enumerate(Q_history[1:]):
                if idx == 0:
                    idcs = np.argwhere(np.nan_to_num(Q_rep, nan=0) != np.nan_to_num(Q_history[0], nan=0)).flatten()
                else:
                    idcs = np.argwhere(np.nan_to_num(Q_history[idx], nan=0) != np.nan_to_num(Q_rep, nan=0)).flatten()

                if len(idcs) == 0:
                    continue
                
                st = idcs[0]
                ac = idcs[1]
                # stas = [idcs[i] for i in range(0, len(idcs), 2)]
                # acts = [idcs[i] for i in range(1, len(idcs), 2)]

                # for seq_idx, (st, ac) in enumerate(zip(stas, acts)):

                fig = plt.figure(figsize=(27, 16), constrained_layout=True)
                ax  = plt.subplot(221)
                plot_maze(ax, Q_rep, agent, move)
                ax1 = plt.subplot(223)
                plot_replay(ax1, agent, [st, ac])
                # ax2 = plt.subplot(222)
                # plot_maze(ax2, gain_history[idx+1], agent)
                # ax3 = plt.subplot(224)
                # plot_need(ax3, need_history[idx+1], agent)
                plt.savefig(os.path.join(save_path, 'move_%s_%u.png')%(file.split('_')[-1][:-4], idx))
                plt.close()

    return None

def plot_env(ax, env):
                
    # state grid
    for st_x in range(env.num_x_states):
        ax.axvline(st_x, c='k', linewidth=0.6)
    for st_y in range(env.num_y_states):
        ax.axhline(st_y, c='k', linewidth=0.6)
    
    for k in env.blocked_state_actions:
        s, a = k
        i, j = env._convert_state_to_coords(s)
        if a == 0:
            ax.hlines((env.num_y_states-i), j, j+1, linewidth=6, color='b')
        elif a == 3:
            ax.vlines(j+1, (env.num_y_states-i)-1, (env.num_y_states-i), linewidth=6, color='b')

    if len(env.nan_states) > 0:
        patches = []
        for s in env.nan_states:
            sy, sx   = env._convert_state_to_coords(s)
            patches += [Rectangle((sx, env.num_y_states-sy-1), 1, 1, edgecolor='k', facecolor='k', linewidth=1)]

        collection = PatchCollection(patches, match_original=True)
        ax.add_collection(collection)

    # goal symbol
    goal_y, goal_x   = env._convert_state_to_coords(env.goal_state)
    ax.scatter(goal_x+0.5, env.num_y_states - goal_y -0.5, s=600, c='orange', marker=r'$\clubsuit$', alpha=0.7)

    ax.set_xlim(0, env.num_x_states)
    ax.set_ylim(0, env.num_y_states)

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

def plot_maze(ax, Q, agent, move=None):
    
    # state grid
    for st_x in range(agent.num_x_states):
        ax.axvline(st_x, c='k', linewidth=0.6)
    for st_y in range(agent.num_y_states):
        ax.axhline(st_y, c='k', linewidth=0.6)

    nan_idcs = np.argwhere(np.all(np.isnan(Q), axis=1)).flatten()
    Q[nan_idcs, :] = 0

    Q_plot = np.zeros(agent.num_states)
    for s in range(agent.num_states):
        max_val = 0
        for a in range(agent.num_actions):
            if ~np.isnan(Q[s, a]):
                if np.absolute(Q[s, a]) > np.absolute(max_val):
                    Q_plot[s] = Q[s, a]
                    max_val   = Q[s, a]
    # Q_plot   = np.nanmax(Q, axis=1).reshape(agent.num_y_states, agent.num_x_states)[::-1, :]
    Q_plot = Q_plot.reshape(agent.num_y_states, agent.num_x_states)[::-1, :]

    if np.all(Q_plot == 0):
        sns.heatmap(np.absolute(Q_plot), cmap=['white'], annot=False, fmt='.2f', cbar=True, vmin=0, vmax=1, ax=ax)
    else:
        sns.heatmap(np.absolute(Q_plot), cmap='Greys', annot=True, fmt='.2f', cbar=True, vmin=0, vmax=1, ax=ax)
    
    # arrows for actions
    patches = []
    for st in np.delete(range(agent.num_states), [agent.goal_state] + agent.nan_states):
        for ac in range(4):
            if ~np.isnan(Q[st, ac]):
                if Q[st, ac] == 0:
                    patches += add_patches(st, ac, 0, agent.num_y_states, agent.num_x_states)
                else:
                    patches += add_patches(st, ac, Q[st, ac], agent.num_y_states, agent.num_x_states)
                # patches += add_patches(st, ac, Q[st, ac], agent.num_y_states, agent.num_x_states)
                for bidx, l in enumerate(agent.uncertain_states_actions):
                    if [st, ac] in l:
                        if agent.barriers[bidx]:
                            i, j = np.argwhere(np.arange(agent.num_states).reshape(agent.num_y_states, agent.num_x_states) == st).flatten()
                            if ac == 0:
                                ax.hlines((agent.num_y_states-i), j, j+1, linewidth=6, color='b')
                            elif ac == 2:
                                ax.vlines(j, (agent.num_y_states-i)-1, (agent.num_y_states-i), linewidth=6, color='b')
                        break

    if len(agent.nan_states) > 0:
        for s in agent.nan_states:
            sy, sx   = agent._convert_state_to_coords(s)
            patches += [Rectangle((sx, agent.num_y_states-sy-1), 1, 1, edgecolor='k', facecolor='k', linewidth=1)]

    collection = PatchCollection(patches, match_original=True)
    ax.add_collection(collection)

    # goal symbol
    goal_y, goal_x   = np.argwhere(np.arange(agent.num_states).reshape(agent.num_y_states, agent.num_x_states) == agent.goal_state).flatten()
    ax.scatter(goal_x+0.5, agent.num_y_states - goal_y -0.5, s=600, c='orange', marker=r'$\clubsuit$', alpha=0.7)

    # agent location
    if move is not None:
        agent_state      = move[-1]
        agent_y, agent_x = np.argwhere(np.arange(agent.num_states).reshape(agent.num_y_states, agent.num_x_states) == agent_state).flatten()
        ax.scatter(agent_x+0.5, agent.num_y_states - agent_y -0.5, s=600, c='green', alpha=0.7)

    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_xlim(0, agent.num_x_states)
    ax.set_ylim(0, agent.num_y_states)

    if move is not None:
        ax.set_title('[' + ' '.join(map(str, [int(i) for i in move])) + ']', fontsize=20)

    return None

def plot_replay(ax, agent, move=None):

    sns.heatmap(np.zeros((agent.num_states, 4)), cmap=['white'], annot=False, fmt='.2f', cbar=True, ax=ax)
    
    # arrows for actions
    patches = []
    for st in np.delete(range(agent.num_states), [agent.goal_state] + agent.nan_states):
        for ac in range(4):
            if len(move) > 0:
                if (st == move[0]) and (ac == move[1]):
                    patches += add_patches(st, ac, 1, agent.num_y_states, agent.num_x_states)
            else:
                pass
            if [st, ac] in [i for j in agent.blocked_state_actions for i in j]:
                i, j = np.argwhere(np.arange(agent.num_states).reshape(agent.num_y_states, agent.num_x_states) == st).flatten()
                if ac == 0:
                    ax.hlines((agent.num_y_states-i), j, j+1, linewidth=6, color='b')
                elif ac == 3:
                    ax.vlines(j+1, (agent.num_y_states-i)-1, (agent.num_y_states-i), linewidth=6, color='b')

    if len(agent.nan_states) > 0:
        for s in agent.nan_states:
            sy, sx   = agent._convert_state_to_coords(s)
            patches += [Rectangle((sx, agent.num_y_states-sy-1), 1, 1, edgecolor='k', facecolor='k', linewidth=1)]

    collection = PatchCollection(patches, match_original=True)
    ax.add_collection(collection)
                
    # state grid
    for st_x in range(agent.num_x_states):
        ax.axvline(st_x, c='k', linewidth=0.6)
    for st_y in range(agent.num_y_states):
        ax.axhline(st_y, c='k', linewidth=0.6)

    goal_y, goal_x   = np.argwhere(np.arange(agent.num_states).reshape(agent.num_y_states, agent.num_x_states) == agent.goal_state).flatten()

    # goal symbol
    ax.scatter(goal_x+0.5, agent.num_y_states - goal_y -0.5, s=600, c='orange', marker=r'$\clubsuit$', alpha=0.7)

    # agent location
    if move is not None:
        agent_state      = move[0]
        agent_y, agent_x = np.argwhere(np.arange(agent.num_states).reshape(agent.num_y_states, agent.num_x_states) == agent_state).flatten()
    
        # agent location
        ax.scatter(agent_x+0.5, agent.num_y_states - agent_y -0.5, s=600, c='green', alpha=0.7)
        ax.set_title('[' + ' '.join(map(str, [int(i) for i in move])) + ']', fontsize=20)

    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_xlim(0, agent.num_x_states)
    ax.set_ylim(0, agent.num_y_states)

    return None

def plot_need(ax, need, agent):
    
    need_plot = need.reshape(agent.num_y_states, agent.num_x_states)[::-1, :]
    need_plot = need_plot/np.nanmax(need_plot)
    if np.all(need_plot == 0):
        sns.heatmap(need_plot, cmap=['white'], annot=False, fmt='.2f', cbar=True, ax=ax)
    else:
        sns.heatmap(need_plot, cmap='Blues', annot=True, fmt='.2f', cbar=True, ax=ax)
    
    # arrows for actions
    patches = []
    for st in np.delete(range(agent.num_states), [agent.goal_state] + agent.nan_states):
        for ac in range(4):
            if [st, ac] in agent.blocked_state_actions:
                i, j = np.argwhere(np.arange(agent.num_states).reshape(agent.num_y_states, agent.num_x_states) == st).flatten()
                if ac == 0:
                    ax.hlines((agent.num_y_states-i), j, j+1, linewidth=6, color='b')
                elif ac == 3:
                    ax.vlines(j+1, (agent.num_y_states-i)-1, (agent.num_y_states-i), linewidth=6, color='b')

    if len(agent.nan_states) > 0:
        for s in agent.nan_states:
            sy, sx   = agent._convert_state_to_coords(s)
            patches += [Rectangle((sx, agent.num_y_states-sy-1), 1, 1, edgecolor='k', facecolor='k', linewidth=1)]

    collection = PatchCollection(patches, match_original=True)
    ax.add_collection(collection)
                
    # state grid
    for st_x in range(agent.num_x_states):
        ax.axvline(st_x, c='k', linewidth=0.6)
    for st_y in range(agent.num_y_states):
        ax.axhline(st_y, c='k', linewidth=0.6)

    # goal symbol
    goal_y, goal_x   = np.argwhere(np.arange(agent.num_states).reshape(agent.num_y_states, agent.num_x_states) == agent.goal_state).flatten()
    ax.scatter(goal_x+0.5, agent.num_y_states - goal_y -0.5, s=600, c='orange', marker=r'$\clubsuit$', alpha=0.7)

    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_xlim(0, agent.num_x_states)
    ax.set_ylim(0, agent.num_y_states)

    ax.set_title('Need', fontsize=20)

    return None