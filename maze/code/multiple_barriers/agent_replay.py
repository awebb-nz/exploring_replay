from environment import Environment
import numpy as np
from copy import deepcopy, copy
import os, shutil, ast

class Agent(Environment):

    def __init__(self, *configs):
        
        '''
        ----
        configs is a list containing 
                    [0] agent parameters and [1] environment parameters

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
        beta_online              -- online inverse temperature
        beta_gain                -- gain inverse temperature
        need_beta                -- need inverse temperature
        policy_type              -- softmax / greedy
        ----
        '''
        
        ag_config  = configs[0]
        env_config = configs[1]
        
        super().__init__(**env_config)
        self._init_barriers(bars=[0, 0, 0])
        
        self.__dict__.update(**ag_config)

        self.state = self.start_state

        # initialise MF Q values
        self._init_q_values()

        # initialise prior
        self.M = np.ones((len(self.barriers), 2))

        return None
        
    def _init_q_values(self):

        self.Q = np.zeros((self.num_states, self.num_actions))

        # set edge Q values to np.nan
        for s in np.delete(range(self.num_states), [self.goal_state] + self.nan_states):
            for a in range(self.num_actions):
                bidx = self._check_uncertain([s, a])
                if bidx is None:
                    s1, _ = self._get_new_state(s, a, unlocked=False)
                    if (s1 == s):
                        self.Q[s, a] = np.nan

        if len(self.nan_states) > 0:
            for s in self.nan_states:
                self.Q[s, :] = np.nan

        if self.env_name == 'tolman1':
            self.Q[8,  1] = np.nan
        elif self.env_name == 'tolman2':
            self.Q[20, 1] = np.nan
        elif self.env_name == 'tolman3':
            self.Q[8,  3] = np.nan
        # elif self.env_name == 'tolman123':
        #     self.Q[8,  1] = np.nan
        #     self.Q[20, 1] = np.nan
        #     self.Q[8,  3] = np.nan
        elif self.env_name == 'u':
            self.Q[1,  2] = np.nan

        self.Q_nans = self.Q.copy()

        return None

    def _solve_mb(self, eps, barriers=None):
        
        if barriers is None:
            barriers = self.barriers
        else:
            self.barriers = barriers

        Q_MB  = self.Q_nans.copy()
        delta = 1
        while delta > eps:
            Q_MB_new = Q_MB.copy()
            for s in np.delete(range(self.num_states), self.goal_state):
                for a in range(self.num_actions):
                    if ~np.isnan(Q_MB[s, a]):
                        bidx = self._check_uncertain([s, a])
                        if bidx is not None:
                            if self.barriers[bidx]:
                                s1, r = self._get_new_state(s, a, unlocked=False)
                            else:
                                s1, r = self._get_new_state(s, a, unlocked=True)
                        else:
                            s1, r = self._get_new_state(s, a, unlocked=True)
                        Q_MB_new[s, a] += r + self.gamma*np.nanmax(Q_MB[s1, :]) - Q_MB_new[s, a]
            diff  = np.abs(Q_MB_new - Q_MB)
            delta = np.nanmax(diff[:])
            Q_MB  = Q_MB_new

        return Q_MB_new

    def _policy(self, q_vals, temp=None):

        '''
        ----
        Agent's policy

        q_vals -- q values at the current state
        ----
        '''
        if np.all(np.isnan(q_vals)):
            return np.full(self.num_actions, 1/self.num_actions)

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

        if self.policy_type == 'softmax':

            if temp is not None:
                t = temp
            else:
                t = self.online_beta

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

        probs_before = self._policy(q_before, temp=self.gain_beta)
        probs_after  = self._policy(q_after, temp=self.gain_beta)

        return np.nansum((probs_after-probs_before)*q_after)

    def _compute_need(self, T, Q):
        
        '''
        ---
        Compute need associated with each state
        ---
        '''

        Ts = np.zeros((self.num_states, self.num_states))
        for s in range(self.num_states):
            probs = self._policy(Q[s, :], temp=self.need_beta)
            for a in range(self.num_actions):
                Ts[s, :] += probs[a]*T[s, a, :]

        return np.linalg.inv(np.eye(self.num_states) - self.gamma*Ts)

    def _find_belief(self, z):

        b = z[0]
        s = z[1]

        for hi in range(self.horizon):
            for _, vals in self.belief_tree[hi].items():
                    
                if (s == vals[0][1]) and np.array_equal(b, vals[0][0]):
                    q = vals[1]

                    return True, q

        return False, None

    def _simulate_trajs(self):

        '''
        ---
        Simulate forward histrories from the agent's current belief
        Important for computing Need of distal beliefs
        ---
        '''

        his = {s:{} for s in range(self.num_states)}

        for sim in range(self.num_sims):
            s     = self.state
            b     = self.M
            d     = 1

            # current state
            bkey = str(b.tolist()).strip()
            if bkey not in his[s].keys():
                his[s][bkey] = [np.full(self.num_sims, np.nan), np.full(self.num_sims, np.nan)]
            
            # need
            his[s][bkey][0][sim] = 1
            # number of steps
            his[s][bkey][1][sim] = 0

            while ((self.gamma**d) > 1e-4):

                check, q = self._find_belief([b, s])

                if not check:
                    break

                qvals = q[s, :]
                probs = self._policy(qvals, temp=self.need_beta)
                a     = np.random.choice(range(self.num_actions), p=probs)
                
                bidx = self._check_uncertain([s, a])
                if bidx is not None:

                    s1u, _ = self._get_new_state(s, a, unlocked=True)
                    s1l, _ = self._get_new_state(s, a, unlocked=False)

                    # sample next state
                    bp = b[bidx, 0]/np.sum(b[bidx, :])
                    
                    s1 = np.random.choice([s1u, s1l], p=[bp, 1-bp])

                    # update belief based on the observed transition
                    b = self._belief_plan_update(b, bidx, s, s1)
                    bkey = str(b.tolist()).strip()

                else:
                    s1, _  = self._get_new_state(s, a, unlocked=False)
                
                if bkey not in his[s1].keys():
                    his[s1][bkey] = [np.full(self.num_sims, np.nan), np.full(self.num_sims, np.nan)]

                curr_val = his[s1][bkey][0][sim]
                if np.isnan(curr_val):
                    his[s1][bkey][0][sim] = self.gamma**d
                    his[s1][bkey][1][sim] = d
                
                s  = s1
                d += 1

        return his

    def _belief_step_update(self, M, idx, s, s1):

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
            M_out[idx, 1] += 1
        # successful
        else:
            M_out[idx, 0] += 1

        return M_out

    def _belief_plan_update(self, M, idx, s, s1):

        M_out = M.copy()

        # unsuccessful 
        if s == s1:
            M_out[idx, 1] = 1
            M_out[idx, 0] = 0
        # successful
        else:
            M_out[idx, 0] = 1
            M_out[idx, 1] = 0

        return M_out

    def _belief_trial_update(self, M, tried=False, success=None):

        M_out = M

        if not tried:
            M_out = self.kappa*self.phi + (1-self.kappa)*M_out
        else:
            if success:
                M_out = 1 - self.kappa*(1-self.phi)
            else:
                M_out = self.kappa*self.phi

        return M_out

    def _check_uncertain(self, sa: list):

        for bidx, l in enumerate(self.uncertain_states_actions):
            if sa in l:
                return bidx
        
        return None

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
                if np.array_equal(vals[0][0], b) and (vals[0][1] == s):
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
        btree[0][idx] = [[self.M.copy(), self.state], self.Q.copy(), []]

        for hi in range(1, self.horizon):
            
            # unique index for each belief
            idx = 0

            if len(btree[hi-1]) == 0:
                break

            for prev_idx, vals in btree[hi-1].items():
                
                # retrieve previous belief information
                b        = vals[0][0].copy()
                s        = vals[0][1]
                q        = vals[1].copy()

                # terminate at the goal state
                if s == self.goal_state:
                    continue
                
                for a in range(self.num_actions):
                    if ~np.isnan(self.Q[s, a]):
                        
                        bidx = self._check_uncertain([s, a])
                        if bidx is not None:

                            # if it's the uncertain state+action then this generates 
                            # two distinct beliefs
                            # first when the agent transitions through
                            s1u, _ = self._get_new_state(s, a, unlocked=True)
                            b1u    = self._belief_plan_update(b, bidx, s, s1u)

                            # second when it doesn't
                            s1l    = s
                            b1l    = self._belief_plan_update(b, bidx, s, s)

                            # check if this belief already exists
                            hiu, idxu, checku = self._check_belief_exists(btree, [b1u, s1u])
                            hil, idxl, checkl = self._check_belief_exists(btree, [b1l, s1l])
                            # if it doesn't exist then add it to the belief tree
                            # and add its key to the previous belief that gave rise to it
                            if not checku and not checkl:
                                btree[hi][idx]            = [[b1u.copy(), s1u], q.copy(), []]
                                btree[hi][idx+1]          = [[b1l.copy(), s1l], q.copy(), []]
                                btree[hi-1][prev_idx][2] += [[[a, hi, idx], [a, hi, idx+1]]]
                                idx                      += 2
                            elif not checku and checkl:
                                btree[hi][idx]            = [[b1u.copy(), s1u], q.copy(), []]
                                btree[hi-1][prev_idx][2] += [[[a, hi, idx], [a, hil, idxl]]]
                                idx                      += 1
                            elif checku and not checkl:
                                btree[hi][idx]            = [[b1l.copy(), s1l], q.copy(), []]
                                btree[hi-1][prev_idx][2] += [[[a, hiu, idxu], [a, hi, idx]]]
                                idx                      += 1
                            else:
                                to_add = [[a, hiu, idxu], [a, hil, idxl]]
                                if (to_add not in btree[hi-1][prev_idx][2]):
                                    btree[hi-1][prev_idx][2] += [to_add]
                            # if the new belief already exists then we just need to add 
                            # the key of that existing belief to the previous belief

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
                                if [a, hip, idxp] not in btree[hi-1][prev_idx][2]:
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
        for s in range(self.num_states):
            for a in range(self.num_actions):
                s1l, _ = self._get_new_state(s, a, unlocked=False)

                bidx = self._check_uncertain([s, a])
                if bidx is not None:

                    s1u, _ = self._get_new_state(s, a, unlocked=True)
                
                    Ta[s, a, s1u] = b[bidx, 0]/np.sum(b[bidx, :])
                    Ta[s, a, s1l] = b[bidx, 1]/np.sum(b[bidx, :])

                else:
                    Ta[s, a, s1l] = 1

        T = np.zeros((self.num_states, self.num_states))
        for s in range(self.num_states):
            qvals = Q[s, :]
            probs = self._policy(qvals, temp=self.online_beta)
            for a in range(self.num_actions):
                T[s, :] += probs[a] * Ta[s, a, :]

        return T

    def _build_pneed_tree(self, ttree):

        '''
        ---
        Compute Need for each information state

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

        for s in ttree.keys():
            
            for k1, vals1 in ttree[s].items():
                        
                b     = np.array(ast.literal_eval(k1))

                hip, idxp, check = self._check_belief_exists(self.belief_tree, [b, s])

                if check:

                    Q     = self.belief_tree[hip][idxp][1].copy()
                    T     = self._get_state_state(b, Q)
                    SR_k  = np.linalg.inv(np.eye(self.num_states) - self.gamma*T)

                    bn      = vals1[0]
                    bp      = vals1[1]

                    maskedn = bn[~np.isnan(bn)]
                    maskedp = bp[~np.isnan(bn)]
                    
                    av_SR   = 0
                    
                    for idx in range(len(maskedn)):
                        
                        SR = SR_k.copy()
                        
                        for i in range(int(maskedp[idx])+1):
                            SR -= (self.gamma**i)*np.linalg.matrix_power(T, i)

                        av_SR += maskedn[idx] + SR[self.state, s]
                    
                    ntree[s] += av_SR/self.num_sims

        return ntree

    def _imagine_update(self, Q_old, state, b, val, btree):
        
        q_old_vals = Q_old[state, :].copy()

        tds = []

        if len(val) == 2:

            a, hiu, idxu = val[0][0], val[0][1], val[0][2]
            _, hil, idxl = val[1][0], val[1][1], val[1][2]

            s1u          = btree[hiu][idxu][0][1]
            q_prime_u    = btree[hiu][idxu][1][s1u, :].copy()

            s1l          = btree[hil][idxl][0][1]
            q_prime_l    = btree[hil][idxl][1][s1l, :].copy()

            y, x = self._convert_state_to_coords(s1u)
            rew  = self.config[y, x]

            tds += [q_old_vals[a] + self.alpha_r*(rew + self.gamma*np.nanmax(q_prime_u) - q_old_vals[a])]
            tds += [q_old_vals[a] + self.alpha_r*(0 + self.gamma*np.nanmax(q_prime_l) - q_old_vals[a])]

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
            bidx = self._check_uncertain([state, a])
            b0   = b[bidx, 0]/np.sum(b[bidx, :])
            b1   = 1 - b[bidx, 0]/np.sum(b[bidx, :])
            q_new_vals[a] = b0*tds[0] + b1*tds[1]

        Q_new[state, :]   = q_new_vals

        return Q_new

    def _generate_forward_sequences(self, updates, pntree):

        seq_updates = []

        for update in updates:
            for l in range(self.max_seq_len-1):
                
                if l == 0:
                    pool = [update]
                else:
                    pool = deepcopy(tmp)

                tmp  = []
                
                for seq in pool: # take an existing sequence
                    
                    lhi   = seq[0][-1]

                    if lhi == (self.horizon - 2):
                        continue

                    lidx      = seq[1][-1]
                    la        = seq[2][-1, 1]
                    next_idcs = self.belief_tree[lhi][lidx][2]

                    tt = []
                    for next_idx in next_idcs:
                        if isinstance(next_idx[0], list):
                            tt += [next_idx[0]]
                            tt += [next_idx[1]]
                        else:
                            tt += [next_idx]
                    next_idcs = tt

                    for next_idx in next_idcs:

                        nhi  = next_idx[1]
                        nidx = next_idx[2]

                        if la == next_idx[0]:

                            vals = self.belief_tree[nhi][nidx]
                            s        = vals[0][1]
                            if s == self.goal_state:
                                continue
                            b        = vals[0][0].copy()

                            if s not in seq[2][:, 0]:
                            
                                Q_old  = vals[1].copy()

                                next_next_idcs = vals[2]

                                for next_next_idx in next_next_idcs:
                                    
                                    this_seq = deepcopy(seq)
                                    Q = this_seq[3].copy()

                                    Q_new  = self._imagine_update(Q_old, s, b, next_next_idx, self.belief_tree)

                                    need   = pntree[s]
                                    gain   = self._compute_gain(Q_old[s, :].copy(), Q_new[s, :].copy())

                                    this_seq[0]  = np.append(this_seq[0], nhi)
                                    this_seq[1]  = np.append(this_seq[1], nidx)
                                    if len(next_next_idx) == 2:
                                        a = next_next_idx[0][0]
                                    else:
                                        a = next_next_idx[0]
                                    this_seq[2]  = np.vstack((this_seq[2], np.array([s, a])))
                                    Q[s, :]      = Q_new[s, :].copy()
                                    this_seq[3]  = Q.copy()
                                    this_seq[4]  = np.append(this_seq[4], gain.copy())
                                    this_seq[5]  = np.append(this_seq[5], need.copy())
                                    this_seq[6]  = np.append(this_seq[6], np.dot(this_seq[4], this_seq[5]))
                                    tmp         += [deepcopy(this_seq)]

                if len(tmp) > 0:
                    seq_updates += tmp

        return seq_updates

    def _generate_reverse_sequences(self, updates, pntree):

        seq_updates = []

        for update in updates:
            for l in range(self.max_seq_len-1):
                
                if l == 0:
                    pool = [update]
                else:
                    pool = deepcopy(tmp)

                tmp  = []
                
                for seq in pool: # take an existing sequence
                    
                    lhi   = seq[0][-1]
                    lidx  = seq[1][-1]

                    for hor in range(self.horizon):

                        for k, vals in self.belief_tree[hor].items():

                            next_idcs = vals[2]

                            if len(next_idcs) == 0:
                                continue

                            for next_idx in next_idcs:

                                if len(next_idx) == 2:
                                    cond = ((next_idx[0][1] == lhi) and (next_idx[0][2] == lidx)) or ((next_idx[0][1] == lhi) and (next_idx[0][2] == lidx))
                                else:
                                    cond = (next_idx[1] == lhi) and (next_idx[2] == lidx)
                                
                                if cond: # found a prev exp

                                    this_seq = deepcopy(seq)
                                    s        = vals[0][1]
                                    b        = vals[0][0].copy()

                                    if s not in this_seq[2][:, 0]:
                                        
                                        # 
                                        nbtree = deepcopy(self.belief_tree)
                                        Q      = seq[3].copy()
                                        nbtree[lhi][lidx][1] = Q.copy()
                                        Q_old  = nbtree[hor][k][1].copy()

                                        Q_new  = self._imagine_update(Q_old, s, b, next_idx, nbtree)

                                        need   = pntree[s]
                                        gain   = self._compute_gain(Q_old[s, :].copy(), Q_new[s, :].copy())
                                        evb    = gain*need

                                        this_seq[0]  = np.append(this_seq[0], hor)
                                        this_seq[1]  = np.append(this_seq[1], k)
                                        if len(next_idx) == 2:
                                            a = next_idx[0][0]
                                        else:
                                            a = next_idx[0]
                                        this_seq[2]  = np.vstack((this_seq[2], np.array([s, a])))
                                        Q[s, :]      = Q_new[s, :].copy()
                                        this_seq[3]  = Q.copy()
                                        this_seq[4]  = np.append(this_seq[4], gain.copy())
                                        this_seq[5]  = np.append(this_seq[5], need.copy())
                                        this_seq[6]  = np.append(this_seq[6], np.dot(this_seq[4], this_seq[5]))
                                        tmp         += [deepcopy(this_seq)]

                if len(tmp) > 0:
                    seq_updates += tmp

        return seq_updates

    def _generate_single_updates(self, pntree):

        updates     = []

        # first generate single-step updates
        for hi in reversed(range(self.horizon-1)):
            if len(self.belief_tree[hi+1]) == 0:
                continue

            for idx, vals in self.belief_tree[hi].items():
                
                state = vals[0][1]

                # do not consider if goal state
                if state == self.goal_state:
                    continue
                
                b     = vals[0][0]
                Q_old = vals[1]

                for val in vals[2]:

                    Q_new = self._imagine_update(Q_old, state, b, val, self.belief_tree)

                    # generalisation -- ?? We need to compute the potential benefit of a single update at <s', b'> at all other beliefs;
                    # that is, <s', b*> for all b* in B. The equation for Need is:
                    # \sum_{<s', b'>} \sum_i \gamma^i P(<s, b> -> <s', b'>, i, \pi_{old})
                    # The equation for Gain is:
                    # \sum_{<s', b'>} \sum_a [\pi_{new}(a | <s', b'>) - \pi_{new}(a | <s', b'>)]q_{\pi_{new}}(<s', b'>, a)

                    need  = pntree[state]
                    gain  = self._compute_gain(Q_old[state, :].copy(), Q_new[state, :].copy())
                    evb   = need * gain

                    if len(val) == 2:
                        a = val[0][0]
                    else:
                        a = val[0]

                    if evb > self.xi:
                        updates += [[np.array([hi]), np.array([idx]), np.array([state, a]).reshape(-1, 2), Q_new.copy(), np.array([gain]), np.array([need]), np.array([evb])]]

        return updates

    def _get_highest_evb(self, updates):
        
        max_evb = 0
        loc     = None
        for idx, upd in enumerate(updates):
            evb = upd[-1][-1]
            if evb > max_evb:
                max_evb = evb
                loc     = idx 

        return loc

    def _replay(self):
        
        Q_history    = [self.Q.copy()]
        gain_history = [None]
        need_history = [None]

        self.belief_tree = self._build_belief_tree()
        traj_tree        = self._simulate_trajs()
        pneed_tree       = self._build_pneed_tree(traj_tree)
        
        num = 1
        while True:
            updates = self._generate_single_updates(pneed_tree)

            if self.sequences:
                rev_updates  = self._generate_reverse_sequences(updates, pneed_tree)
                fwd_updates  = self._generate_forward_sequences(updates, pneed_tree)
                if len(rev_updates) > 0:
                    updates += rev_updates
                if len(fwd_updates) > 0:
                    updates += fwd_updates

            idx = self._get_highest_evb(updates)

            if idx is None:
                break

            evb = updates[idx][-1]

            if evb[-1] < self.xi:
                break
            else:

                hi  = updates[idx][0]
                k   = updates[idx][1]
                
                s  = updates[idx][2][:, 0]
                a  = updates[idx][2][:, 1]
                
                Q_new = updates[idx][3].copy()

                gain  = updates[idx][4]
                need  = updates[idx][5]

                for sidx, si in enumerate(s):
                    Q_old = self.belief_tree[hi[sidx]][k[sidx]][1].copy()
                    b     = self.belief_tree[hi[sidx]][k[sidx]][0][0]
                    print('%u - Replay %u/%u [<%u, (%.2f, %.2f, %.2f)>, %u] horizon %u, q_old: %.2f, q_new: %.2f, gain: %.2f, need: %.2f, evb: %.2f'%(num, sidx+1, len(s), si, b[0, 0]/b[0, :].sum(), b[1, 0]/b[1, :].sum(), b[2, 0]/b[2, :].sum(), a[sidx], hi[sidx], Q_old[si, a[sidx]], Q_new[si, a[sidx]], gain[sidx], need[sidx], evb[sidx]), flush=True)
                    Q_old[si, a[sidx]] = Q_new[si, a[sidx]].copy()
                    self.belief_tree[hi[sidx]][k[sidx]][1] = Q_old.copy()
            
                    if np.array_equal(b, self.M):
                        self.Q[si, a[sidx]] = Q_new[si, a[sidx]].copy()
                        Q_history          += [self.Q.copy()]
                        gain_history       += [np.sum(gain)]
                        need_history       += [np.sum(need)]

                traj_tree      = self._simulate_trajs()
                pneed_tree     = self._build_pneed_tree(traj_tree)

                num += 1

        return Q_history, gain_history, need_history

    def run_simulation(self, num_steps=100, save_path=None):

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
        else:
            self.save_path = None

        replay = False
        num_replay_moves = 0

        for move in range(num_steps):
            
            s      = self.state

            # choose action and receive feedback
            probs  = self._policy(self.Q[s, :], temp=self.online_beta)
            a      = np.random.choice(range(self.num_actions), p=probs)

            bidx = self._check_uncertain([s, a])
            if bidx is not None:
                if self.barriers[bidx]:
                    s1, r  = self._get_new_state(s, a, unlocked=False)
                else:
                    s1, r  = self._get_new_state(s, a, unlocked=True)
                
                # update belief
                self.M = self._belief_plan_update(self.M, bidx, s, s1)

                if replay:
                    # fetch Q values of the new belief
                    for hi in range(self.horizon):
                        for k, vals in self.belief_tree[hi].items():
                            b = vals[0][0]
                            if np.array_equal(self.M, b):
                                state  = vals[0][1]
                                Q_vals = self.belief_tree[hi][k][1]
                                self.Q[state, :] = Q_vals[state, :].copy()

            else:
                s1, r  = self._get_new_state(s, a, unlocked=True)

            q_old  = self.Q[s, a]

            # update MF Q values
            self._qval_update(s, a, r, s1)

            print('Move %u/%u, [<%u, [%.2f, %.2f, %.2f]>, %u], q_old: %.2f, q_new: %.2f\n'%(move, num_steps, s, self.M[0, 0]/self.M[0, :].sum(), self.M[1, 0]/self.M[1, :].sum(), self.M[2, 0]/self.M[2, :].sum(), a, q_old, self.Q[s, a]), flush=True)

            # transition to new state
            self.state = s1

            if self.state == self.goal_state:
                replay = True

            if replay:
                Q_history, gain_history, need_history = self._replay()
                num_replay_moves += 1
                if num_replay_moves >= 20:
                    return None

            if save_path:
                if replay:
                    np.savez(os.path.join(save_path, 'Q_%u.npz'%move), barrier=self.barriers, Q_history=Q_history, M=self.M, gain_history=gain_history, need_history=need_history, move=[s, a, r, s1])
                else:
                    np.savez(os.path.join(save_path, 'Q_%u.npz'%move), barrier=self.barriers, Q_history=self.Q, M=self.M, move=[s, a, r, s1])

            if s1 == self.goal_state:
                self.state = self.start_state

        return None
