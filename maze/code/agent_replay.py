from environment import Environment
import numpy as np
import os, shutil

class Agent(Environment):

    def __init__(self, config, start_coords, goal_coords, blocked_state_actions, uncertain_states_actions, alpha, alpha_r, gamma, horizon, xi, num_sims, policy_temp=None, policy_type='softmax'):
        
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
        num_sims                 -- number of simulations for need estimation 
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
        self.num_sims    = num_sims

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
        self.M  = np.ones(2)

        # beta prior for reward magnitude
        self.Mr = np.ones(2)

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

        his = {s:{} for s in range(self.num_states)}

        for sim in range(self.num_sims):
            s     = self.state
            b     = self.M.copy()
            d     = 1

            # current state
            if (int(b[0]), int(b[1])) not in his[s].keys():
                his[s][(int(b[0]), int(b[1]))] = [np.full(self.num_sims, np.nan), np.full(self.num_sims, np.nan)]
            
            # need
            his[s][(int(b[0]), int(b[1]))][0][sim] = 1
            # number of steps
            his[s][(int(b[0]), int(b[1]))][1][sim] = 0

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

                    # update belief based on the observed transition
                    b = self._belief_update(b, s, s1)

                else:
                    s1, _  = self._get_new_state(s, a, unlocked=False)
                
                if (int(b[0]), int(b[1])) not in his[s1].keys():
                    his[s1][(int(b[0]), int(b[1]))] = [np.full(self.num_sims, np.nan), np.full(self.num_sims, np.nan)]

                curr_val = his[s1][(int(b[0]), int(b[1]))][0][sim]
                if np.isnan(curr_val):
                    his[s1][(int(b[0]), int(b[1]))][0][sim] = self.gamma**d
                    his[s1][(int(b[0]), int(b[1]))][1][sim] = d
                
                s  = s1
                d += 1

        return his

    def _belief_update(self, M, s, s1):

        '''
        ----
        Bayesian belief updates for transition beta prior

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

        # if s != s1:
        self.Q[s, a] += self.alpha*(r + self.gamma*np.nanmax(self.Q[s1, :]) - self.Q[s, a])
        # else:
            # self.Q[s, a] += self.alpha*(0 - self.Q[s, a])

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
                            # first is when the agent transitions through
                            s1u, _ = self._get_new_state(s, a, unlocked=True)
                            b1u    = self._belief_update(b, s, s1u)

                            # second is when it doesn't
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

        '''
        ---
        Marginalise T[s, a, s'] over actions with the current policy 

        b -- current belief about transition structure
        Q -- MF Q values associated with this belief
        ---
        '''
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

        ntree = np.zeros(self.num_states)

        for hi in reversed(range(self.horizon)):
            if len(btree[hi]) == 0:
                continue

            for _, vals in btree[hi].items():
                
                b     = vals[0][0]
                state = vals[0][1]
                Q     = vals[1]
                T     = self._get_state_state(b, Q)
                SR_k  = np.linalg.inv(np.eye(self.num_states) - self.gamma*T)

                for k1, vals1 in ttree[state].items():
                    
                    if not np.array_equal(b, np.array(k1, dtype=type(b[0].item()))):
                        continue

                    bn      = vals1[0]
                    bp      = vals1[1]

                    maskedn = bn[~np.isnan(bn)]
                    maskedp = bp[~np.isnan(bn)]
                    
                    av_SR   = 0
                    
                    for idx in range(len(maskedn)):
                        
                        SR = SR_k.copy()
                        
                        for i in range(int(maskedp[idx])+1):
                            SR -= (self.gamma**i)*np.linalg.matrix_power(T, i)

                        av_SR += maskedn[idx] + SR[self.start_state, state]
                    
                    ntree[state] += av_SR/self.num_sims

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
                    need = pntree[state]
                    gain = self._compute_gain(q_old_vals, q_new_vals)
                    evb  = need * gain

                    if self.save_path is not None:
                        self.file.write('\nHorizon %u, [<%u, [%u, %u]>, %u], q_old: %.2f, q_new: %.2f, gain: %.4f, need: %.4f, evb: %.5f'%(hi, state, b[0], b[1], a, q_old_vals[a], q_new_vals[a], gain, need, evb))
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
                if self.save_path is not None:
                    self.file.write('\n\nReplay [<%u, [%u, %u]>, %u] horizon %u, belief , q_old: %.2f, q_new: %.2f, evb: %.5f\n'%(s, b[0], b[1], a, hi, Q_old[s, a], Q_new[s, a], max_evb))
                new_stuff             = belief_tree[hi][k[0]]
                new_stuff[1]          = Q_new.copy()
                belief_tree[hi][k[0]] = new_stuff
                traj_tree             = self._simulate_trajs(belief_tree)
                pneed_tree            = self._build_pneed_tree(belief_tree, traj_tree)

                if hi == 0:

                    Q_new = np.zeros((self.num_states, self.num_actions))
                    for k, v in belief_tree[0].items():
                        s = v[0][1]
                        Q = v[1]
                        Q_new[s, :] = Q[s, :]

                    Q_history     += [Q_new]
                    gain_history  += [gain]
                    need_history  += [need]

                #     belief_tree    = self._build_belief_tree()
                #     traj_tree      = self._simulate_trajs(belief_tree)

                #     pneed_tree     = self._build_pneed_tree(belief_tree, traj_tree)
        
        Q_new = np.zeros((self.num_states, self.num_actions))
        for k, v in belief_tree[0].items():
            s = v[0][1]
            Q = v[1]
            Q_new[s, :] = Q[s, :]
        self.Q = Q_new.copy()

        return Q_history, gain_history, need_history

    def run_simulation(self, num_steps=100, start_replay=100, reset_prior=True, save_path=None):

        '''
        ---
        Main loop for the simulation

        num_steps    -- number of simulation steps
        start_replay -- after which step to start replay
        reset_pior   -- whether to reset transition prior before first replay bout
        save_path    -- path for saving data after replay starts
        ---
        '''

        if save_path:
            self.save_path = save_path

            if os.path.isdir(self.save_path):
                shutil.rmtree(self.save_path)
            os.makedirs(self.save_path)

            self.file = open(os.path.join(self.save_path, 'info.txt'), 'w')
            self.file.write('--- Simulation details ---\n')
        else:
            self.save_path = None

        replay  = False
        counter = 0

        for step in range(num_steps):
            
            print('Step %u/%u'%(step+1, num_steps))
            print('Counter %u'%counter)

            s      = self.state

            # choose action and receive feedback
            probs  = self._policy(self.Q[s, :])
            a      = np.random.choice(range(self.num_actions), p=probs)
            s1, r  = self._get_new_state(s, a)

            q_old  = self.Q[s, a]
            # update MF Q values
            self._qval_update(s, a, r, s1)

            if (self.save_path is not None) and replay:
                self.file.write('\n\nMove %u/%u, [<%u, [%u, %u]> %u], q_old: %.2f, q_new: %.2f\n'%(step+1, num_steps, s, self.M[0], self.M[1], a, q_old, self.Q[s, a]))

            # update transition probability belief
            if (s == self.uncertain_states_actions[0]) and (a==self.uncertain_states_actions[1]):
                self.M = self._belief_update(self.M, s, s1)

            # transition to new state
            self.state = s1

            if step == start_replay:

                replay = True

                if reset_prior:
                    self.M = np.ones(2)

            if replay:

                Q_history, gain_history, need_history = self._replay()

                if save_path:
                    np.savez(os.path.join(save_path, 'Q_%u.npz'%step), Q_history=Q_history, gain_history=gain_history, need_history=need_history, move=[s, a, r, s1])

            if s1 == self.goal_state:
                self.state = self.start_state

        if self.save_path is not None:
            self.file.close()

        return None