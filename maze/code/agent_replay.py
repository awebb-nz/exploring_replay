from environment import Environment
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

    def _find_belief(self, btree, z):

        b = z[0]
        s = z[1]

        for hi in range(self.horizon):
            for _, vals in btree[hi].items():
                    
                if (s == vals[0][1]) and np.array_equal(b, vals[0][0]):
                    q = vals[1]

                    return True, q

        return False, None

    def _simulate_trajs(self, btree):

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

            # current state
            if (int(b[0]), int(b[1])) not in his[s].keys():
                his[s][(int(b[0]), int(b[1]))] = [np.full(num_sims, np.nan), np.full(num_sims, np.nan)]
            
            # can be reached in 0 steps
            his[s][(int(b[0]), int(b[1]))][0][sim] = 0
            # with probability 1
            his[s][(int(b[0]), int(b[1]))][1][sim] = 1

            while ((self.gamma**d) > 1e-4):

                check, q = self._find_belief(btree, [b, s])

                if not check:
                    break

                qvals = q[s, :]
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
                    b = self._belief_update(b, s, s1)

                else:
                    s1, _  = self._get_new_state(s, a, unlocked=False)
                    proba *= 1*probs[a]
                
                if (int(b[0]), int(b[1])) not in his[s1].keys():
                    his[s1][(int(b[0]), int(b[1]))] = [np.full(num_sims, np.nan), np.full(num_sims, np.nan)]

                curr_val = his[s1][(int(b[0]), int(b[1]))][0][sim]
                if (np.isnan(curr_val)) or (curr_val > d):
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

                maskedn   = bn[~np.isnan(bn)]
                maskedp   = bp[~np.isnan(bn)]

                av_steps = int(np.ceil(maskedn.mean()))
                P        = np.sum(maskedp) / len(maskedp)

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

    def _qval_update(self, s, a, r, s1):

        '''
        ----
        MF Q values update

        s           -- previous state 
        a           -- chosen action
        r           -- received reward
        s1          -- resulting new state
        ----
        ''' 

        if s != s1:
            self.Q[s, a] += self.alpha*(r + self.gamma*np.nanmax(self.Q[s1, :]) - self.Q[s, a])
        else:
            self.Q[s, a] += self.alpha*(0 - self.Q[s, a])

        return None

    def _check_belief_exists(self, btree, z):

        '''
        ---
        Checks if the belief state z already exists in the tree
        ---
        '''

        b = z[0]
        s = z[1]

        for hi in range(self.horizon):
            this_tree = btree[hi]
            for k, vals in this_tree.items():
                if (np.array_equal(vals[0][0], b)) and (vals[0][1] == s):
                    return hi, k, True

        return None, None, False

    def _build_belief_tree(self):
        
        '''
        ---
        Build a tree with future belief states up to horizon self.horizon
        ---
        '''

        # each horizon hosts a number of belief states
        btree = {hi:{} for hi in range(self.horizon)}

        # create a merged tree -- one single tree for all information states
        idx = 0
        for s in np.delete(range(self.num_states), self.goal_state):
            btree[0][idx] = [[self.M.copy(), s], self.Q.copy(), []]
            idx          += 1

        for hi in range(1, self.horizon):
            
            # unique index for each belief
            idx = 0

            for k, vals in btree[hi-1].items():
                
                # retrieve previous belief information
                b        = vals[0][0]
                s        = vals[0][1]
                q        = vals[1]
                prev_idx = k

                # terminate at the goal state
                if s == self.goal_state:
                    continue
                
                for a in range(self.num_actions):
                    if ~np.isnan(self.Q[s, a]):

                        if (s == self.uncertain_states_actions[0]) and (a==self.uncertain_states_actions[1]):
                            
                            # if it's the uncertain state+action then this generates two distinct beliefs
                            s1u, _ = self._get_new_state(s, a, unlocked=True)
                            b1u    = self._belief_update(b, s, s1u)

                            s1l    = s
                            b1l    = self._belief_update(b, s, s)

                            # check if this belief already exists
                            hip, idxp, check = self._check_belief_exists(btree, [b1u, s1u])
                            # if it doesn't exist then add it to the belief tree
                            # and add its key to the previous belief that gave rise to it
                            if not check:
                                btree[hi][idx]            = [[b1u.copy(), s1u], q.copy(), []]
                                btree[hi][idx+1]          = [[b1l.copy(), s1l], q.copy(), []]
                                btree[hi-1][prev_idx][2] += [[[a, hi, idx], [a, hi, idx+1]]]
                                idx                      += 2

                            # if the new belief already exists then we just need to add 
                            # the key of that existing belief to the previous belief
                            else:
                                btree[hi-1][prev_idx][2] += [[a, hip, idxp]]

                        else:
                            s1u, _ = self._get_new_state(s, a, unlocked=False)
                            b1u    = b.copy()

                            # check if this belief already exists
                            hip, idxp, check = self._check_belief_exists(btree, [b1u, s1u])
                            # if it doesn't exist then add it to the belief tree
                            # and add its key to the previous belief that gave rise to it
                            if not check:
                                btree[hi][idx]            = [[b1u.copy(), s1u], q.copy(), []]
                                btree[hi-1][prev_idx][2] += [[a, hi, idx]]
                                idx                      += 1
                            # if the new belief already exists then we just need to add 
                            # the key of that existing belief to the previous belief
                            else:
                                btree[hi-1][prev_idx][2] += [[a, hip, idxp]]

        return btree

    def _get_state_state(self, b, Q):

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
            qvals = Q[s, :]
            probs = self._policy(qvals)
            for a in range(self.num_actions):
                T[s, :] += probs[a] * Ta[s, a, :]

        return T

    def _build_pneed_tree(self, btree, ttree):

        '''
        ---
        Compute Need for each information state

        btree -- tree with information states
        ttree -- tree with the estimated probabilities 
        ---
        '''

        # here is the picture:
        #
        #                -
        #               / 
        #              X
        #             / \
        #            /   -
        # A - - - - -
        #            \   -
        #             \ /
        #              -
        #               \
        #                -
        #
        # A is the agent's current state
        # X is the belief at which an update is executed
        # 
        # The path to X is estimated based on 
        # monte-carlo returns in the method called 
        # simulate_trajs()

        ntree  = {hi:{} for hi in range(self.horizon)}
        
        for hi in reversed(range(self.horizon)):
            if len(btree[hi]) == 0:
                continue

            for k, vals in btree[hi].items():
                
                b     = vals[0][0]
                state = vals[0][1]
                Q     = vals[1]

                if (int(b[0]), int(b[1])) not in ttree[state].keys():
                    ntree[hi][k] = 0
                else:
                    these_vals = ttree[state][(int(b[0]), int(b[1]))]
                    proba      = these_vals[1]
                    N          = these_vals[0]

                    T  = self._get_state_state(b, Q)
                    SR = np.linalg.inv(np.eye(self.num_states) - self.gamma*T)

                    for i in range(N+1):
                        SR -= (self.gamma**i)*np.linalg.matrix_power(T, i)

                    SR += (self.gamma**N)*proba

                    ntree[hi][k] = SR[self.state, state]

        return ntree

    def _get_updates(self, btree, pntree):

        nbtree    = {hi:{} for hi in range(self.horizon)}
        evb_tree  = {hi:{} for hi in range(self.horizon)}
        need_save = np.zeros((self.num_states))
        gain_save = np.full((self.num_states, self.num_actions), np.nan)

        for hi in reversed(range(self.horizon-1)):
            if len(btree[hi+1]) == 0:
                continue

            for k, vals in btree[hi].items():

                b     = vals[0][0]
                state = vals[0][1]
                Q_old = vals[1]

                if state == self.goal_state:
                    continue

                q_old_vals = Q_old[state, :].copy()

                for _, val in enumerate(vals[2]):

                    tds = []

                    if len(val) == 2:

                        a, hil, idxl = val[0][0], val[0][1], val[0][2]
                        _, hiu, idxu = val[1][0], val[1][1], val[1][2]

                        s1l          = btree[hil][idxl][0][1]
                        q_prime_u    = btree[hil][idxl][1][s1l, :].copy()
                        s1u          = btree[hiu][idxu][0][1]
                        q_prime_l    = btree[hil][idxl][1][s1u, :].copy()

                        y, x = self._convert_state_to_coords(s1u)
                        rew  = self.config[y, x]

                        tds += [q_old_vals[a] + self.alpha_r*(rew + self.gamma*np.nanmax(q_prime_u) - q_old_vals[a])]
                        tds += [q_old_vals[a] + self.alpha_r*(0 - q_old_vals[a])]

                    else:
                        a, hi1, idx1 = val[0], val[1], val[2]
                        s1           = btree[hi1][idx1][0][1]
                        q_prime      = btree[hi1][idx1][1][s1, :].copy()

                        y, x = self._convert_state_to_coords(s1)
                        rew  = self.config[y, x]
                        tds += [q_old_vals[a] + self.alpha_r*(rew + self.gamma*np.nanmax(q_prime) - q_old_vals[a])]

                    # get the new (updated) q value
                    Q_new      = Q_old.copy()
                    q_new_vals = q_old_vals.copy()

                    if len(tds) != 2: 
                        q_new_vals[a] = tds[0]
                    else:    
                        b0 = b[0]/np.sum(b)
                        b1 = b[1]/np.sum(b)
                        q_new_vals[a] = b0*tds[0] + b1*tds[1]
                    
                    Q_new[state, :]     = q_new_vals

                    new_key             = tuple([k, a])
                    nbtree[hi][new_key] = [[b, state], Q_new]

                    # generalisation -- ?? We need to compute the potential effect of a single update at <s', b'> on all other beliefs;
                    # that is, <s', b*> for all b* in B. The equation for Need is:
                    # \sum_{<s', b'>} \sum_i \gamma^i P(<s, b> -> <s', b'>, i, \pi_{old})
                    # The equation for Gain is:
                    # \sum_{<s', b'>} \sum_a [\pi_{new}(a | <s', b'>) - \pi_{new}(a | <s', b'>)]q_{\pi_{new}}(<s', b'>, a)
                    need = pntree[hi][k]
                    gain = self._compute_gain(q_old_vals, q_new_vals)
                    evb  = need * gain

                    evb_tree[hi][new_key] = evb

                    if hi == 0:
                        gain_save[state, a] = gain
                        need_save[state]    = need

        return nbtree, evb_tree, gain_save, need_save

    def _get_highest_evb(self, evb_tree):
        
        max_evb = 0
        for hi in range(self.horizon):
            if len(evb_tree[hi]) == 0:
                continue
            for k, evb in evb_tree[hi].items():
                if evb > max_evb:
                    max_evb = evb
                    hir     = hi
                    kr      = k

        if max_evb == 0:
            max_evb = self.xi - 1
            hir     = None
            kr      = None

        return hir, kr, max_evb

    def _replay(self):
        
        Q_history    = [self.Q.copy()]
        gain_history = [None]
        need_history = [None]

        belief_tree  = self._build_belief_tree()
        traj_tree    = self._simulate_trajs(belief_tree)

        pneed_tree   = self._build_pneed_tree(belief_tree, traj_tree)
        
        while True:
            nbelief_tree, evb_tree, gain, need = self._get_updates(belief_tree, pneed_tree)

            hi, k, max_evb = self._get_highest_evb(evb_tree)
            if max_evb < self.xi:
                break
            else:
                s = nbelief_tree[hi][k][0][1]
                b = nbelief_tree[hi][k][0][0]
                a = k[1]
                
                Q_old = belief_tree[hi][k[0]][1].copy()
                Q_new = nbelief_tree[hi][k][1].copy()

                # if hi == 0:
                print('Replay', [s, a, hi], b, 'Q old: ', np.round(Q_old[s, a], 2), 'Q new: ', np.round(Q_new[s, a], 2), 'EVB: ', np.round(max_evb, 5))

                new_stuff             = belief_tree[hi][k[0]]
                new_stuff[1]          = Q_new.copy()
                belief_tree[hi][k[0]] = new_stuff
                traj_tree             = self._simulate_trajs(belief_tree)
                pneed_tree            = self._build_pneed_tree(belief_tree, traj_tree)

                if hi == 0:

                    self.Q[s, a]   = Q_new[s, a]
                    Q_history     += [self.Q.copy()]
                    gain_history  += [gain]
                    need_history  += [need]

                    belief_tree    = self._build_belief_tree()
                    traj_tree      = self._simulate_trajs(belief_tree)

                    pneed_tree     = self._build_pneed_tree(belief_tree, traj_tree)

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

            s      = self.state
            probs  = self._policy(self.Q[s, :])
            a      = np.random.choice(range(self.num_actions), p=probs)
            s1, r  = self._get_new_state(s, a)

            self._qval_update(s, a, r, s1)

            if (s == self.uncertain_states_actions[0]) and (a==self.uncertain_states_actions[1]):
                self.M = self._belief_update(self.M, s, s1)

            self.state = s1

            if step == 3000:

                replay = True
                self.M = np.ones(2)

                self.Q[16, 3] = 0
                self.Q[16, 2] = 0
                self.Q[16, 1] = 0
                self.Q[17, 0] = 0
                self.Q[17, 1] = 0
                self.Q[17, 2] = 0
                self.Q[22, 0] = 0
                self.Q[22, 2] = 0
                self.Q[22, 3] = 0
                self.Q[23, 0] = 0
                self.Q[23, 2] = 0

            if replay:

                counter += 1
                if counter == 50:
                    return None

                Q_history, gain_history, need_history = self._replay()

                if save_path:
                    np.savez(os.path.join(save_path, 'Q_%u.npz'%step), Q_history=Q_history, gain_history=gain_history, need_history=need_history, move=[s, a, r, s1])

            if s1 == self.goal_state:
                self.state = self.start_state

        return None