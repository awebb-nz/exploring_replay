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
        self._init_env()
        
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
                check = True
                if [s, a] in self.uncertain_states_actions:
                    check = False
                    continue
                if check:
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
                        if [s, a] in self.uncertain_states_actions:
                            idx   = self.uncertain_states_actions.index([s, a])
                            if self.barriers[idx]:
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
            b     = self.M
            d     = 1

            # current state
            if str(b.tolist()).strip() not in his[s].keys():
                his[s][str(b.tolist()).strip()] = [np.full(self.num_sims, np.nan), np.full(self.num_sims, np.nan)]
            
            # need
            his[s][str(b.tolist()).strip()][0][sim] = 1
            # number of steps
            his[s][str(b.tolist()).strip()][1][sim] = 0

            while ((self.gamma**d) > 1e-4):

                check, q = self._find_belief(btree, [b, s])

                if not check:
                    break

                qvals = q[s, :]
                probs = self._policy(qvals, temp=self.need_beta)
                a     = np.random.choice(range(self.num_actions), p=probs)
                
                if [s, a] in self.uncertain_states_actions:

                    idx = self.uncertain_states_actions.index([s, a])
                    
                    s1u, _ = self._get_new_state(s, a, unlocked=True)
                    s1l, _ = self._get_new_state(s, a, unlocked=False)

                    # sample next state
                    bp = b[idx, 0]/np.sum(b[idx, :])
                        
                    s1 = np.random.choice([s1u, s1l], p=[bp, 1-bp])

                    # update belief based on the observed transition
                    b = self._belief_step_update(b, idx, s, s1)

                else:
                    s1, _  = self._get_new_state(s, a, unlocked=False)
                
                if str(b.tolist()).strip() not in his[s1].keys():
                    his[s1][str(b.tolist()).strip()] = [np.full(self.num_sims, np.nan), np.full(self.num_sims, np.nan)]

                curr_val = his[s1][str(b.tolist()).strip()][0][sim]
                if np.isnan(curr_val):
                    his[s1][str(b.tolist()).strip()][0][sim] = self.gamma**d
                    his[s1][str(b.tolist()).strip()][1][sim] = d
                
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
        for s in np.delete(range(self.num_states), self.nan_states):
            btree[0][idx] = [[self.M, s], self.Q.copy(), []]
            idx          += 1

        for hi in range(1, self.horizon):
            
            # unique index for each belief
            idx = 0

            for k, vals in btree[hi-1].items():
                
                # retrieve previous belief information
                b        = vals[0][0].copy()
                s        = vals[0][1]
                q        = vals[1]
                prev_idx = k

                # terminate at the goal state
                if s == self.goal_state:
                    continue
                
                for a in range(self.num_actions):
                    if ~np.isnan(self.Q[s, a]):

                        if [s, a] in self.uncertain_states_actions:
                            
                            bidx   = self.uncertain_states_actions.index([s, a])
                            # if it's the uncertain state+action then this generates 
                            # two distinct beliefs
                            # first when the agent transitions through
                            s1u, _ = self._get_new_state(s, a, unlocked=True)
                            b1u    = self._belief_step_update(b, bidx, s, s1u)

                            # second when it doesn't
                            s1l    = s
                            b1l    = self._belief_step_update(b, bidx, s, s)

                            # check if this belief already exists
                            hip, idxp, check = self._check_belief_exists(btree, [b1u, s1u])
                            # if it doesn't exist then add it to the belief tree
                            # and add its key to the previous belief that gave rise to it
                            if not check:
                                btree[hi][idx]            = [[b1u, s1u], q.copy(), []]
                                btree[hi][idx+1]          = [[b1l, s1l], q.copy(), []]
                                btree[hi-1][prev_idx][2] += [[[a, hi, idx], [a, hi, idx+1]]]
                                idx                      += 2

                            # if the new belief already exists then we just need to add 
                            # the key of that existing belief to the previous belief
                            else:
                                btree[hi-1][prev_idx][2] += [[a, hip, idxp]]

                        else:
                            s1u, _ = self._get_new_state(s, a, unlocked=False)
                            b1u    = b

                            # check if this belief already exists
                            hip, idxp, check = self._check_belief_exists(btree, [b1u, s1u])
                            # if it doesn't exist then add it to the belief tree
                            # and add its key to the previous belief that gave rise to it
                            if not check:
                                btree[hi][idx]            = [[b1u, s1u], q.copy(), []]
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
        for s in range(self.num_states):
            for a in range(self.num_actions):
                s1l, _ = self._get_new_state(s, a, unlocked=False)

                if [s, a] in self.uncertain_states_actions:
                    
                    bidx   = self.uncertain_states_actions.index([s, a])

                    s1u, _ = self._get_new_state(s, a, unlocked=True)
                
                    Ta[s, a, s1u] = b[bidx, 0]/np.sum(b[bidx, :])
                    Ta[s, a, s1l] = 1 - b[bidx, 0]/np.sum(b[bidx, :])

                else:
                    Ta[s, a, s1l] = 1

        T = np.zeros((self.num_states, self.num_states))
        for s in range(self.num_states):
            qvals = Q[s, :]
            probs = self._policy(qvals, temp=self.online_beta)
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
                    
                    if not np.array_equal(b, np.array(ast.literal_eval(k1))):
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

    def _imagine_update(self, btree, vals):

        b     = vals[0][0]
        state = vals[0][1]
        Q_old = vals[1]

        q_old_vals = Q_old[state, :].copy()

        for val in vals[2]:

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
                idx = self.uncertain_states_actions.index([state, a])
                b0  = b[idx, 0]/np.sum(b[idx, :])
                b1  = 1 - b[idx, 0]/np.sum(b[idx, :])
                q_new_vals[a] = b0*tds[0] + b1*tds[1]

        Q_new[state, :]   = q_new_vals

        return state, b, a, Q_new

    def _get_updates(self, btree, pntree):

        updates     = []
        evbs        = []
        seq_updates = []
        seq_evbs    = []

        max_seq_len = 4

        # first generate single-step updates
        for hi in reversed(range(self.horizon-1)):
            if len(btree[hi+1]) == 0:
                continue

            for idx, vals in btree[hi].items():
                
                state = vals[0][1]
                # do not consider if goal state
                if state == self.goal_state:
                    continue

                state, b, a, Q_new = self._imagine_update(btree, vals)

                updates += [[hi, idx, [state, b, a], Q_new]]

                # generalisation -- ?? We need to compute the potential benefit of a single update at <s', b'> at all other beliefs;
                # that is, <s', b*> for all b* in B. The equation for Need is:
                # \sum_{<s', b'>} \sum_i \gamma^i P(<s, b> -> <s', b'>, i, \pi_{old})
                # The equation for Gain is:
                # \sum_{<s', b'>} \sum_a [\pi_{new}(a | <s', b'>) - \pi_{new}(a | <s', b'>)]q_{\pi_{new}}(<s', b'>, a)
                Q_old = vals[1]

                need  = pntree[state]
                gain  = self._compute_gain(Q_old[state, :].copy(), Q_new[state, :].copy())
                evb   = need * gain

                evbs += [evb]

                # here we can elongate this experience
                if hi == 0:

                    for l in range(max_seq_len):
                        
                        if l == 0:
                            pool = [[updates[-1][0], updates[-1][1], np.array([updates[-1][2][0], updates[-1][2][2]], dtype=int).reshape(1, 2), updates[-1][3]]]
                        else:
                            pool = deepcopy(tmp)

                        tmp  = []
                        
                        for seq in pool: # take an existing sequence
                            Q_new = seq[3]
                            idx   = seq[1]
                            # here need to find an exp to elongate with
                            # search through the btree to find all exps 
                            # that lead to the one of interest

                            for k, vals in btree[0].items():
                                
                                next_idcs = vals[2]

                                if len(next_idcs) == 0:
                                    continue

                                for next_idx in next_idcs:

                                    if (next_idx[1] == 0) and (next_idx[2] == idx): # found a prev exp
                                        
                                        # i can just do the update here since
                                        # only 1st horizon gets updates
                                        nbtree = deepcopy(btree)
                                        nbtree[hi][idx][1] = Q_new.copy()

                                        state, b, a, Q_new = self._imagine_update(nbtree, vals)

                                        if a == next_idx[0]:
                                            this_seq     = deepcopy(seq)
                                            this_seq[1]  = next_idx[1]
                                            if state not in this_seq[2][:, 0]:
                                                this_seq[2]  = np.concatenate((this_seq[2], np.array([state, a]).reshape(1, 2))).reshape(-1, 2)
                                                this_seq[3]  = deepcopy(Q_new)
                                                tmp         += [deepcopy(this_seq)]
                                            
                        if len(tmp) > 0:
                            seq_updates += tmp

        print('w8 here')


        return updates, evbs

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
        
        Q_history      = [self.Q.copy()]
        gain_history   = [None]
        need_history   = [None]

        belief_tree    = self._build_belief_tree()
        traj_tree      = self._simulate_trajs(belief_tree)

        pneed_tree     = self._build_pneed_tree(belief_tree, traj_tree)
        
        while True:
            updates, evbs = self._get_updates(belief_tree, pneed_tree)

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
                print('Replay [<%u, %u>] horizon %u, q_old: %.2f, q_new: %.2f, evb: %.5f'%(s, a, hi, Q_old[s, a], Q_new[s, a], max_evb), flush=True)

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

        replay  = True
        tried   = False
        success = None

        for move in range(num_steps):
            
            s      = self.state

            # choose action and receive feedback
            probs  = self._policy(self.Q[s, :], temp=self.online_beta)
            a      = np.random.choice(range(self.num_actions), p=probs)

            if [s, a] in self.uncertain_states_actions:
                idx    = self.uncertain_states_actions.index([s, a])

                if self.barriers[idx]:
                    s1, r  = self._get_new_state(s, a, unlocked=False)
                else:
                    s1, r  = self._get_new_state(s, a, unlocked=True)
            else:
                s1, r  = self._get_new_state(s, a, unlocked=True)

            q_old  = self.Q[s, a]
            # update MF Q values
            self._qval_update(s, a, r, s1)

            print('Move %u/%u, [<%u, %.2f> %u], q_old: %.2f, q_new: %.2f\n'%(move, num_steps, s, self.M, a, q_old, self.Q[s, a]), flush=True)

            # update transition probability belief
            if [s, a] in self.uncertain_states_actions:
                self.M = self._belief_step_update(self.M, idx, s, s1)

            # transition to new state
            self.state = s1

            if replay:
                Q_history, gain_history, need_history = self._replay()

            if save_path:
                if replay:
                    np.savez(os.path.join(save_path, 'Q_%u.npz'%move), barrier=self.barriers, Q_history=Q_history, M=self.M, gain_history=gain_history, need_history=need_history, move=[s, a, r, s1])
                else:
                    np.savez(os.path.join(save_path, 'Q_%u.npz'%move), barrier=self.barriers, Q_history=self.Q, M=self.M, move=[s, a, r, s1])

            if s1 == self.goal_state:
                self.state = self.start_state

        return None
