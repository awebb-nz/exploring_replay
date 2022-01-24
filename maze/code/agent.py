from sre_parse import State
import numpy as np
from copy import deepcopy
import os, shutil

class Environment:

    def __init__(self, config, blocked_state_actions: list, start_state, goal_state):

        '''
        ----
        config                -- matrix which specifies the env
        blocked_state_actions -- list with state-action pairs [s, a] which are blocked
        ----
        '''

        self.config                = config
        self.blocked_state_actions = blocked_state_actions
        self.num_x_states          = config.shape[1]
        self.num_y_states          = config.shape[0]

        self.num_states            = self.num_x_states*self.num_y_states
        self.num_actions           = 4

        self.start_state           = start_state
        self.goal_state            = goal_state

        return None

    def _get_new_state(self, s, a, unlocked=False):

        '''
        ----
        s        -- current state of the agent
        a        -- chosen action
        unlocked -- whether the action is available or not (for blocked_state_actions)
        ----
        '''

        if s == self.goal_state:
            return self.start_state, 0

        y_coord, x_coord = self._convert_state_to_coords(s)

        # ----
        # first consider edge cases
        # at the top and choose up
        case1 = (y_coord == 0) and (a == 0)
        # at the bottom and choose down
        case2 = (y_coord == self.num_y_states - 1) and (a == 1)
        # at the left edge and choose left
        case3 = (x_coord == 0) and (a == 2)
        # at the right edge and choose right
        case4 = (x_coord == self.num_x_states - 1) and (a == 3)

        if case1 or case2 or case3 or case4:
            r = self.config[y_coord, x_coord]
            return s, r
        else:
            # ----
            # choose up
            if a == 0:
                x1_coord, y1_coord = x_coord, y_coord - 1
            # choose down
            elif a == 1:
                x1_coord, y1_coord = x_coord, y_coord + 1
            # choose left 
            elif a == 2:
                x1_coord, y1_coord = x_coord - 1, y_coord
            # choose right
            else:
                x1_coord, y1_coord = x_coord + 1, y_coord

            # check the barriers
            if (unlocked == True) or ([s, a] not in self.blocked_state_actions):
                r  = self.config[y1_coord, x1_coord]
                s1 = self._convert_coords_to_state([y1_coord, x1_coord])
                return s1, r
            else:
                r = self.config[y_coord, x_coord]
                return s, r
            

    def _convert_state_to_coords(self, s):

        y_coord = s // self.num_x_states
        x_coord = s % self.num_x_states

        return [y_coord, x_coord]

    def _convert_coords_to_state(self, coords: list):

        y_coord, x_coord = coords
        states = np.arange(self.num_states).reshape(self.num_y_states, self.num_x_states)

        return states[y_coord, x_coord]
        

class Agent(Environment):

    def __init__(self, config, start_state, goal_state, blocked_state_actions, uncertain_state_coords, uncertain_action, alpha, gamma, horizon, xi, policy_temp=None, policy_type='softmax'):
        
        '''
        ----
        config                 -- matrix which specifies the env
        start_coords           -- start state coordinates
        goal_coords            -- goal state coordinates 
        blocked_state_actions  -- list with state-action pairs [s, a] which are blocked
        uncertain_state_coords -- blockage which the agent is uncertain about
        uncertain_action       -- ''-''
        alpha                  -- value learning rate
        gamma                  -- discount factor
        horizon                -- planning / replay horizon
        xi                     -- replay EVB threshold
        policy_temp            -- inverse temperature
        policy_type            -- softmax / greedy
        ----
        '''
        
        super().__init__(config, blocked_state_actions, start_state, goal_state)

        self.Q = np.zeros((self.num_states, self.num_actions))
        # beta prior for the uncertain transition
        self.M = np.ones(2)

        # transition matrix for other transitions
        self.T = np.zeros((self.num_states, self.num_actions, self.num_states))
        for s in np.delete(range(self.num_states), self.goal_state):
            for a in range(self.num_actions):
                s1, _ = self._get_new_state(s, a)
                self.T[s, a, s1] = 1
        for a in range(self.num_actions):
            self.T[self.goal_state, a, self.start_state] = 1

        self.uncertain_state  = self._convert_coords_to_state(uncertain_state_coords)
        self.uncertain_action = uncertain_action

        self.policy_temp = policy_temp
        self.policy_type = policy_type
        self.alpha       = alpha
        self.gamma       = gamma
        self.horizon     = horizon
        self.xi          = xi

        self.state       = self.start_state

        return None
        
    def _policy(self, q_values):

        '''
        ----
        Agent's policy

        q_values -- q values at the current state
        temp     -- inverse temperature
        type     -- softmax / greeedy
        ----
        '''

        if np.all(q_values == q_values.max()):
            return np.ones(self.num_actions)/self.num_actions

        if self.policy_temp:
            t = self.policy_temp
        else:
            t = 1
            
        if self.policy_type == 'softmax':
            return np.exp(q_values*t)/np.sum(np.exp(q_values*t))
        elif self.policy_type == 'greedy':
            if np.all(q_values == q_values.max()):
                a        = np.random.choice(range(self.num_actions), p=np.ones(self.num_actions)/self.num_actions)
                probs    = np.zeros(self.num_actions)
                probs[a] = 1
                return probs
            else:
                return np.array(q_values >= q_values.max()).astype(int)
        else:
            raise KeyError('Unknown policy type')

    def _compute_gain(self, q_before, q_after):

        probs_before = self._policy(q_before)
        probs_after  = self._policy(q_after)

        return np.dot((probs_after-probs_before), q_after)

    def _compute_need(self, T, Q):

        Ts = np.zeros((self.num_states, self.num_states))
        for s in range(self.num_states):
            probs = self._policy(Q[s, :])
            for a in range(self.num_actions):
                Ts[s, :] += probs[a]*T[s, a, :]

        return np.linalg.inv(np.eye(self.num_states) - self.gamma*Ts)

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
        qvals[a] += self.alpha*(r + self.gamma*np.max(Q[s1, :]) - qvals[a])

        return qvals

    def _build_belief_tree(self, s):

        btree = {hi:{} for hi in range(self.horizon)}
        btree[0][(None, s, 0, 0)] = self.M

        for hi in range(1, self.horizon):
            c = 0
            for k, b in btree[hi-1].items():
                prev_c  = k[-1]
                prev_s1 = k[1]
                for a in range(self.num_actions):
                    if (prev_s1 == self.uncertain_state) and (a == self.uncertain_action):
                        s1, _    = self._get_new_state(prev_s1, a, unlocked=True)
                        b1       = self._belief_update(b, prev_s1, s1)
                        btree[hi][(a, s1, prev_c, c)] = b1
                        c       += 1
                        s1, _    = self._get_new_state(prev_s1, a, unlocked=False)
                        b1       = self._belief_update(b, prev_s1, s1)
                        btree[hi][(a, s1, prev_c, c)] = b1
                        c       += 1
                    else:
                        s1, _    = self._get_new_state(prev_s1, a)
                        btree[hi][(a, s1, prev_c, c)] = b
                        c       += 1

        return btree

    def _build_qval_tree(self, btree):

        qtree = {hi:{} for hi in range(self.horizon)}

        for hi in range(self.horizon):
            for k, _ in btree[hi].items():
                qtree[hi][k] = self.Q.copy()

        return qtree

    def _build_need_tree(self, btree, qtree):

        ntree = {hi:{} for hi in range(self.horizon)}

        for hi in reversed(range(self.horizon)):
            for k, b in btree[hi].items():

                prev_c = k[-2]
                a      = k[0]
                proba  = 1

                for hin in reversed(range(hi)):
                    for kn, bn in btree[hin].items():
                        if kn[-1] == prev_c:
                            
                            state  = kn[1]
                            Q_vals = qtree[hin][kn].copy()
                            q_vals = Q_vals[state, :]

                            policy_proba = self._policy(q_vals)
                            # if it's the uncertain action & state
                            if (state == self.uncertain_state) and (a == self.uncertain_action):
                                # if successful
                                s1, _ = self._get_new_state(state, a, unlocked=True)
                                if k[1] == s1:
                                    bc = bn[0]/np.sum(bn)
                                else:
                                    bc = bn[1]/np.sum(bn)
                            else:
                                bc = 1
                            proba *= policy_proba[a]*bc

                            prev_c = kn[-2]
                            a      = kn[0]
                            break

                ntree[hi][k] = proba
                        
        return ntree

    def _get_updates(self, btrees, ntrees, qtrees):

        nqval_trees = [{hi:{} for hi in range(self.horizon)} for s in range(self.num_states)]
        evb_trees   = [{hi:{} for hi in range(self.horizon)} for s in range(self.num_states)]
        for s in range(self.num_states):
            btree      = btrees[s]
            qtree      = qtrees[s]
            ntree      = ntrees[s]

            for hi in reversed(range(self.horizon-1)):
                for k, b in btree[hi].items():
                    
                    state = k[1]
                    c     = k[-1]
                    
                    Q_old      = qtree[hi][k].copy()
                    q_old_vals = Q_old[state, :].copy()
                    v_old      = np.dot(self._policy(q_old_vals), q_old_vals)
                    
                    for a in range(self.num_actions):
                        tds = []
                        for k1, Q_prime in qtree[hi+1].items():
                            
                            prev_c = k1[-2]
                            s1     = k1[1]
                            q_prime_vals  = Q_prime[s1, :].copy()
                            
                            if (prev_c == c) and (k1[0] == a):
                                y, x = self._convert_state_to_coords(s1)
                                rew  = self.config[y, x]
                                tds += [rew + self.gamma*np.max(q_prime_vals)]

                                # if it's the uncertain (s, a) pair then this generates 2 belief states
                                if (state == self.uncertain_state) and (a == self.uncertain_action):
                                    if len(tds) == 2:
                                        break
                                # otherwise there's only one possible next state
                                else: 
                                    break

                        # get the new (updated) q value
                        Q_new      = Q_old.copy()
                        q_new_vals = q_old_vals.copy()
                        if len(tds) == 2:
                            b0 = b[0]/np.sum(b)
                            b1 = 1 - b[1]/np.sum(b)
                            q_new_vals[a] = b0*tds[0] + b1*tds[1]
                        else:
                            q_new_vals[a] = tds[0]

                        Q_new[state, :]   = q_new_vals
                        new_key = tuple(list(k) + [a])
                        nqval_trees[s][hi][new_key] = Q_new

                        # v_new   = np.dot(self._policy(q_new_vals), q_new_vals)
                        # evb     = ntree[hi][k] * (v_new - v_old)

                        gain = self._compute_gain(q_old_vals, q_new_vals)


                        T     = self.T.copy()
                        T[self.uncertain_state, self.uncertain_action, :] = np.zeros(self.num_states)
                        s1l   = self._get_new_state(self.uncertain_state, self.uncertain_action, unlocked=False)
                        s1u   = self._get_new_state(self.uncertain_state, self.uncertain_action, unlocked=True)
                        T[self.uncertain_state, self.uncertain_action, s1u] = b[0]/np.sum(b)
                        T[self.uncertain_state, self.uncertain_action, s1l] = (1-b[0]/np.sum(b))
                        need  = self._compute_need(T, Q_old)
                        pneed = ntree[hi][k]
                        # if (state == self.uncertain_state) and (a == self.uncertain_action):
                            # print(hi, new_key, pneed, need[self.state, state], gain)
                        evb_trees[s][hi][new_key] = pneed * gain * need[self.state, state]

        return nqval_trees, evb_trees

    def _replay(self):
        
        Q_history    = [self.Q.copy()]

        belief_trees = []
        qval_trees   = []
        need_trees   = []
        for s in range(self.num_states):
            belief_tree   = self._build_belief_tree(s)
            belief_trees += [belief_tree]
            qval_tree     = self._build_qval_tree(belief_tree)
            qval_trees   += [qval_tree]
            need_tree     = self._build_need_tree(belief_tree, qval_tree)
            need_trees   += [need_tree] 
        
        while True:
            new_qval_trees, evb_trees = self._get_updates(belief_trees, need_trees, qval_trees)
            max_evb = 0
            idx     = []
            for s in range(self.num_states):
                evb_tree = evb_trees[s]
                for hi in range(self.horizon):
                    for k, evb in evb_tree[hi].items():
                        if evb > max_evb:
                            max_evb = evb
                            idx     = [s, hi, k]
                            Q_new   = new_qval_trees[s][hi][k]

            if max_evb < self.xi:
                break
            else:
                print('Replay', idx)
                s, hi, k = idx[0], idx[1], idx[2]
                qval_trees[s][hi][k[:-1]] = Q_new
                need_trees[s] = self._build_need_tree(belief_trees[s], qval_trees[s])

                if hi == 0:
                    self.Q = Q_new
                    Q_history   += [self.Q.copy()] 

                    qval_trees   = []
                    need_trees   = []
                    for s in range(self.num_states):
                        belief_tree   = belief_trees[s]
                        qval_tree     = self._build_qval_tree(belief_tree)
                        qval_trees   += [qval_tree]
                        need_tree     = self._build_need_tree(belief_tree, qval_tree)
                        need_trees   += [need_tree] 

        for s in range(self.num_states):
            for _, v in qval_trees[s][0].items():
                self.Q[s, :] = v[s, :]
        return Q_history

    def run_simulation(self, num_steps=100, save_path=None):

        if save_path:
            if os.path.isdir(save_path):
                shutil.rmtree(save_path)
            os.makedirs(save_path)

        for step in range(num_steps):
            
            print('Step %u/%u'%(step+1, num_steps))

            probs = self._policy(self.Q[self.state, :])
            a     = np.random.choice(range(self.num_actions), p=probs)
            s1, r = self._get_new_state(self.state, a)

            self.Q[self.state, :] = self._qval_update(self.Q, self.state, a, r, s1)
            
            # check if attempted the shortcut
            if (self.state == self.uncertain_state) and (a == self.uncertain_action):
                self.M = self._belief_update(self.M, self.state, s1)

            if step > 100:
                Q_history = self._replay()

                if save_path:
                    np.savez(os.path.join(save_path, 'Q_%u.npz'%step), Q_history=Q_history, move=[self.state, a, r, s1])

            if s1 == self.goal_state:
                self.state = self.start_state
            else:
                self.state = s1

        return None
