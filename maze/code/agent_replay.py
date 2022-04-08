from environment import Environment
from dijkstra import Graph, dijkstra
import numpy as np
import os, shutil

class Agent(Environment):

    def __init__(self, config, start_coords, goal_coords, blocked_state_actions, uncertain_states_actions, alpha, alpha_r, gamma, horizon, xi, policy_temp=None, policy_type='softmax'):
        
        '''
        ----
        config                   -- matrix which specifies the env
        start_coords             -- start state coordinates
        goal_coords              -- goal state coordinates 
        blocked_state_actions    -- list with state-action pairs [s, a] which are blocked
        uncertain_states_actions -- list with states and actions about which the agent becomes uncertain 
        alpha                    -- on-line value learning rate
        alpha_r                  -- replay learning rate
        gamma                    -- discount factor
        horizon                  -- planning / replay horizon
        xi                       -- replay EVB threshold
        policy_temp              -- inverse temperature
        policy_type              -- softmax / greedy
        ----
        '''
        
        super().__init__(config, blocked_state_actions, start_coords, goal_coords)

        self.start_state = self._convert_coords_to_state(start_coords)
        self.goal_state  = self._convert_coords_to_state(goal_coords)

        self.uncertain_states_actions = uncertain_states_actions

        self.policy_temp = policy_temp
        self.policy_type = policy_type
        self.alpha       = alpha
        self.alpha_r     = alpha_r
        self.gamma       = gamma
        self.horizon     = horizon
        self.xi          = xi

        self.state       = self.start_state

        # initialise MF Q values
        self.Q = np.zeros((self.num_states, self.num_actions))

        # set edge Q values to np.nan
        for s in np.delete(range(self.num_states), self.goal_state):
            for a in range(self.num_actions):
                check = True
                if (s == self.uncertain_states_actions[0]):
                    if (a == self.uncertain_states_actions[1]):
                        s1, _ = self._get_new_state(s, a, unlocked=True)
                        check = False
                        continue
                if check:
                    s1, _ = self._get_new_state(s, a, unlocked=False)
                    if s1 == s:
                        self.Q[s, a] = np.nan

        self.Q_nans = self.Q.copy()

        # beta prior for the uncertain transition
        self.M = np.ones(2)

        self.env_graph = Graph()
        # for the dijkstra -- represent the env as a graph
        for s in range(self.num_states):
            self.env_graph.addNode(str(s))
        
        for s in range(self.num_states):
            for a in range(self.num_actions):
                s1, _ = self._get_new_state(s, a, unlocked=False)
                self.env_graph.addEdge(str(s), str(s1), 1)

        return None
        
    def _policy(self, q_vals):

        '''
        ----
        Agent's policy

        q_vals -- q values at the current state
        ----
        '''

        nan_idcs = np.argwhere(np.isnan(q_vals)).flatten()
        if len(nan_idcs) > 0:
            q_vals_allowed = np.delete(q_vals, nan_idcs)
        else:
            q_vals_allowed = q_vals

        if np.all(q_vals_allowed == q_vals_allowed.max()):
            probs = np.ones(len(q_vals_allowed))/len(q_vals_allowed)
            if len(nan_idcs) > 0:
                for nan_idx in nan_idcs:
                    probs = np.insert(probs, nan_idx, 0)
            return probs

        if self.policy_temp:
            t = self.policy_temp
        else:
            t = 1
            
        if self.policy_type == 'softmax':
            probs = np.exp(q_vals_allowed*t)/np.sum(np.exp(q_vals_allowed*t))
            if len(nan_idcs) > 0:
                for nan_idx in nan_idcs:
                    probs = np.insert(probs, nan_idx, 0)
            return probs
        elif self.policy_type == 'greedy':
            if np.all(q_vals_allowed == q_vals_allowed.max()):
                ps           = np.ones(self.num_actions)
                ps[nan_idcs] = 0
                ps          /= ps.sum()
                a            = np.random.choice(range(self.num_actions), p=ps)
                probs        = np.zeros(self.num_actions)
                probs[a]     = 1
                return probs
            else:
                probs = np.zeros(self.num_actions)
                probs[np.nanargmax(q_vals)] = 1
                return probs
        else:
            raise KeyError('Unknown policy type')

    def _compute_gain(self, q_before, q_after):

        '''
        ---
        Compute gain associated with each replay update
        ---
        '''

        probs_before = self._policy(q_before)
        probs_after  = self._policy(q_after)

        return np.nansum((probs_after-probs_before)*q_after)

    def _compute_need(self, T, Q):
        
        '''
        ---
        Compute need associated with each state
        ---
        '''

        Ts = np.zeros((self.num_states, self.num_states))
        for s in range(self.num_states):
            probs = self._policy(Q[s, :])
            for a in range(self.num_actions):
                Ts[s, :] += probs[a]*T[s, a, :]

        return np.linalg.inv(np.eye(self.num_states) - self.gamma*Ts)


    def _simulate_trajs(self):

        '''
        ---
        Simulate forward histrories from the agent's current belief
        Important for computing Need of distal beliefs
        ---
        '''

        his       = {s:{} for s in range(self.num_states)}
        num_sims  = 1000

        for sim in range(num_sims):
            s     = self.state
            b     = self.M.copy()
            d     = 1
            proba = 1
            while ((self.gamma**d) > 0.01):
                
                qvals = self.Q[s, :]
                probs = self._policy(qvals)
                a     = np.random.choice(range(self.num_actions), p=probs)
                
                if (s == self.uncertain_states_actions[0]) and (a == self.uncertain_states_actions[1]):
                    
                    s1u, _ = self._get_new_state(s, a, unlocked=True)
                    s1l, _ = self._get_new_state(s, a, unlocked=False)

                    # sample next state
                    s1 = np.random.choice([s1u, s1l], p=[b[0]/np.sum(b), b[1]/np.sum(b)])
                    
                    bn = b/b.sum()
                    if s1 == s1u:
                        proba *= bn[0]*probs[a]
                    else:
                        proba *= bn[1]*probs[a]

                    # update belief based on the observed transition
                    b  = self._belief_update(b, s, s1)

                else:
                    s1, _  = self._get_new_state(s, a, unlocked=False)
                    proba *= 1*probs[a]
                
                if (int(b[0]), int(b[1])) not in his[s1].keys():
                    his[s1][(int(b[0]), int(b[1]))] = [np.zeros(num_sims), np.zeros(num_sims)]

                curr_val = his[s1][(int(b[0]), int(b[1]))][0][sim]
                if (curr_val == 0) or (curr_val > d):
                    his[s1][(int(b[0]), int(b[1]))][0][sim] = d
                    his[s1][(int(b[0]), int(b[1]))][1][sim] = proba
                
                s  = s1
                d += 1

        # now convert these counts into estimated 
        # probability and distance

        out_tree = {s:{} for s in range(self.num_states)}

        for s in range(self.num_states):
            d  = his[s]
            for b, counts in d.items():
                
                bn       = counts[0]
                bp       = counts[1]

                maskedn   = bn[bn > 0]
                maskedp   = bp[bn > 0]

                av_steps = int(np.ceil(maskedn.mean()))
                P        = np.sum(maskedp) / num_sims

                out_tree[s][int(b[0]), int(b[1])] = [av_steps, P]

        return out_tree

    def _belief_update(self, M, s, s1):

        '''
        ----
        Bayesian belief updates for beta prior

        s           -- previous state 
        a           -- chosen action
        s1          -- resulting new state
        ----
        ''' 

        M_out = M.copy()

        # unsuccessful 
        if s == s1:
            M_out[1] += 1
        # successful
        else:
            M_out[0] += 1

        return M_out

    def _qval_update(self, Q, s, a, r, s1):

        '''
        ----
        MF Q values update

        s           -- previous state 
        a           -- chosen action
        r           -- received reward
        s1          -- resulting new state
        ----
        ''' 

        qvals     = Q[s, :].copy()
        qvals[a] += self.alpha*(r + self.gamma*np.nanmax(Q[s1, :]) - qvals[a])

        return qvals

    def _build_belief_tree(self, s):

        btree = {hi:{} for hi in range(self.horizon)}

        # don't plan at the goal state
        if s == self.goal_state:
            return btree

        btree[0][(None, s, 0, 0)] = self.M

        for hi in range(1, self.horizon):
            c = 0
            for k, b in btree[hi-1].items():
                prev_c  = k[-1]
                prev_s1 = k[1]

                # terminate at the goal state
                if prev_s1 == self.goal_state:
                    continue
                
                for a in range(self.num_actions):
                    if ~np.isnan(self.Q[prev_s1, a]):
                        if (prev_s1 == self.uncertain_states_actions[0]) and (a==self.uncertain_states_actions[1]):
                            s1u, _    = self._get_new_state(prev_s1, a, unlocked=True)
                            b1        = self._belief_update(b, prev_s1, s1u)
                            btree[hi][(a, s1u, prev_c, c)] = b1
                            c        += 1

                            b1        = self._belief_update(b, prev_s1, prev_s1)
                            btree[hi][(a, prev_s1, prev_c, c)] = b1
                            c        += 1
                        else:
                            s1u, _    = self._get_new_state(prev_s1, a, unlocked=False)
                            b1        = b.copy()
                            btree[hi][(a, s1u, prev_c, c)] = b1
                            c        += 1

        return btree

    def _build_qval_tree(self, btree):

        qtree = {hi:{} for hi in range(self.horizon)}

        for hi in range(self.horizon):
            if len(btree[hi]) == 0:
                continue
            for k, _ in btree[hi].items():
                # if (hi == 0) or (hi == (self.horizon - 1)):
                qtree[hi][k] = self.Q.copy()
                # else:
                    # qtree[hi][k] = self.Q_nans.copy()

        return qtree

    def _get_state_state(self, b):

        Ta     = np.zeros((self.num_states, self.num_actions, self.num_states))
        for st in range(self.num_states):
            for at in range(self.num_actions):
                s1l, _ = self._get_new_state(st, at, unlocked=False)

                if (st == self.uncertain_states_actions[0]) and (at == self.uncertain_states_actions[1]):
                    
                    s1u, _ = self._get_new_state(st, at, unlocked=True)
                
                    Ta[st, at, s1u] = b[0]/np.sum(b)
                    Ta[st, at, s1l] = b[1]/np.sum(b)

                else:
                    Ta[st, at, s1l] = 1

        T = np.zeros((self.num_states, self.num_states))
        for s in range(self.num_states):
            qvals = self.Q[s, :]
            probs = self._policy(qvals)
            for a in range(self.num_actions):
                T[s, :] += probs[a] * Ta[s, a, :]

        return T

    def _build_pneed_tree(self, btree, ttree):

        # here is the picture:
        #
        #                -
        #               / 
        #              X
        #             / \
        #            /   -
        # A - - - - R
        #            \   -
        #             \ /
        #              -
        #               \
        #                -
        #
        # A is the agent's current state
        # R is the root state of the tree
        # X is the belief at which an update is executed
        #
        # we want to compute Need based on i) the agent 
        # first reaching the information state X; ii) 
        # considering future branches of the tree; and 
        # iii) what happens in the future beyond the 
        # horizon (resorting to certainty-equivalence)

        ntree  = {hi:{} for hi in range(self.horizon)}
        
        for hi in reversed(range(self.horizon)):
            if len(btree[hi]) == 0:
                continue

            for k, b in btree[hi].items():

                state = k[1] 

                if (int(b[0]), int(b[1])) not in ttree[state].keys():
                    ntree[hi][k] = 0
                else:
                    these_vals = ttree[state][(int(b[0]), int(b[1]))]
                    proba      = these_vals[1]
                    N          = these_vals[0]

                    T  = self._get_state_state(b)
                    SR = np.linalg.inv(np.eye(self.num_states) - self.gamma*T)

                    for i in range(N):
                        SR -= (self.gamma**i)*np.linalg.matrix_power(T, i)

                    SR += (self.gamma**N)*proba
                    ntree[hi][k] = SR[self.state, state]
                    # print(SR[self.state, state])

        return ntree

    def _get_updates(self, btrees, pntrees, qtrees):

        nqval_trees = [{hi:{} for hi in range(self.horizon)} for _ in range(self.num_states)]
        evb_trees   = [{hi:{} for hi in range(self.horizon)} for _ in range(self.num_states)]
        need_save   = np.zeros((self.num_states))
        gain_save   = np.full((self.num_states, self.num_actions), np.nan)

        self._simulate_trajs()

        for s in range(self.num_states):
            btree      = btrees[s]
            qtree      = qtrees[s]
            pntree     = pntrees[s]

            for hi in reversed(range(self.horizon-1)):
                if len(btree[hi+1]) == 0:
                    continue
                for k, b in btree[hi].items():
                    state = k[1]

                    if state == self.goal_state:
                        continue

                    c          = k[-1]

                    Q_old      = qtree[hi][k].copy()
                    q_old_vals = Q_old[state, :].copy()
                    
                    for a in range(self.num_actions):
                        
                        tds = []

                        if ~np.isnan(self.Q[state, a]):
                            
                            for k1, Q_prime in qtree[hi+1].items():
                                
                                prev_c = k1[-2]
                                s1     = k1[1]
                                
                                q_prime_vals = Q_prime[s1, :].copy() 
                                
                                if (prev_c == c) and (k1[0] == a):
                                    y, x = self._convert_state_to_coords(s1)
                                    rew  = self.config[y, x]
                                    tds += [q_old_vals[a] + self.alpha_r*(rew + self.gamma*np.nanmax(q_prime_vals) - q_old_vals[a])]

                                    # if it's the uncertain (s, a) pair then this generates 2 belief states
                                    if (state == self.uncertain_states_actions[0]) and (a==self.uncertain_states_actions[1]): 
                                        if len(tds) == 2:
                                            break
                                    else:
                                        break
                                else:
                                    pass

                            # get the new (updated) q value
                            Q_new      = Q_old.copy()
                            q_new_vals = q_old_vals.copy()

                            if len(tds) != 2: 
                                q_new_vals[a] = tds[0]
                            else:    
                                b0 = b[0]/np.sum(b)
                                b1 = b[1]/np.sum(b)
                                q_new_vals[a] = b0*tds[0] + b1*tds[1]
                            
                            Q_new[state, :]   = q_new_vals
                            new_key = tuple(list(k) + [a])
                            nqval_trees[s][hi][new_key] = Q_new

                            need = pntree[hi][k]
                            gain = self._compute_gain(q_old_vals, q_new_vals)
                            
                            evb  = need * gain

                            evb_trees[s][hi][new_key] = evb

                            if hi == 0:
                                gain_save[state, a] = gain
                                need_save[state]    = need

        return nqval_trees, evb_trees, gain_save, need_save

    def _replay(self):
        
        Q_history    = [self.Q.copy()]
        gain_history = [None]
        need_history = [None]

        belief_trees = []
        qval_trees   = []
        pneed_trees  = []

        traj_tree    = self._simulate_trajs()

        for s in range(self.num_states):
            belief_tree   = self._build_belief_tree(s)
            belief_trees += [belief_tree]
            qval_tree     = self._build_qval_tree(belief_tree)
            qval_trees   += [qval_tree]

            pneed_tree    = self._build_pneed_tree(belief_tree, traj_tree)
            pneed_trees  += [pneed_tree] 
        
        while True:
            new_qval_trees, evb_trees, gain, need = self._get_updates(belief_trees, pneed_trees, qval_trees)
            max_evb = 0
            idx     = []
            for s in range(self.num_states):
                evb_tree = evb_trees[s]
                for hi in range(self.horizon):
                    if len(evb_tree[hi]) == 0:
                        continue
                    for k, evb in evb_tree[hi].items():
                        if evb > max_evb:
                            max_evb = evb
                            idx     = [s, hi, k]
                            Q_new   = new_qval_trees[s][hi][k]

            if max_evb < self.xi:
                break
            else:
                s, hi, k = idx[0], idx[1], idx[2]

                if hi == 0:
                    print('Replay', idx, 'Q old: ', np.round(qval_trees[s][hi][k[:-1]][s, k[-1]], 3), 'Q new: ', np.round(new_qval_trees[s][hi][k][s, k[-1]], 3), 'EVB: ', np.round(max_evb, 3))

                qval_trees[s][hi][k[:-1]] = Q_new
                traj_tree      = self._simulate_trajs()
                pneed_trees[s] = self._build_pneed_tree(belief_trees[s], traj_tree)

                if hi == 0:

                    self.Q        = Q_new
                    Q_history    += [self.Q.copy()]
                    gain_history += [gain]
                    need_history += [need]

                    qval_trees    = []
                    pneed_trees   = []

                    traj_tree     = self._simulate_trajs()

                    for s in range(self.num_states):
                        belief_tree  = belief_trees[s]
                        qval_tree    = self._build_qval_tree(belief_tree)
                        qval_trees  += [qval_tree]
                        pneed_tree   = self._build_pneed_tree(belief_tree, traj_tree)
                        pneed_trees += [pneed_tree] 

        for s in range(self.num_states):
            for _, v in qval_trees[s][0].items():
                self.Q[s, :] = v[s, :]
        return Q_history, gain_history, need_history

    def run_simulation(self, num_steps=100, save_path=None):

        if save_path:
            if os.path.isdir(save_path):
                shutil.rmtree(save_path)
            os.makedirs(save_path)

        replay  = False
        counter = 0

        for step in range(num_steps):
            
            print('Step %u/%u'%(step+1, num_steps))
            print('Counter %u'%counter)

            s = self.state
            probs  = self._policy(self.Q[s, :])
            a      = np.random.choice(range(self.num_actions), p=probs)
            s1, r  = self._get_new_state(s, a) 

            self.Q[self.state, :] = self._qval_update(self.Q, s, a, r, s1)
            self.M = self._belief_update(self.M, s, s1)

            self.state = s1

            if step == 4000:
                self.Q[16, 1] = 0
                self.Q[16, 2] = 0
                self.Q[16, 3] = 0

                self.Q[15, 3] = 0

                self.Q[17, 0] = 0
                self.Q[17, 1] = 0
                self.Q[17, 2] = 0

                self.Q[22, 0] = 0
                self.Q[22, 2] = 0
                self.Q[22, 3] = 0

                self.Q[21, 3] = 0

                self.Q[23, 0] = 0
                self.Q[23, 2] = 0

                replay = True
                self.M = np.ones(2)

            if replay:

                counter += 1
                if counter == 12:
                    return None

                Q_history, gain_history, need_history = self._replay()

                if save_path:
                    np.savez(os.path.join(save_path, 'Q_%u.npz'%step), Q_history=Q_history, gain_history=gain_history, need_history=need_history, move=[s, a, r, s1])

            if s1 == self.goal_state:
                self.state = self.start_state

        return None