import numpy as np
from copy import deepcopy

class Tree:

    def __init__(self, **p):
        
        '''
        ----
        MAB agent

        root_belief   -- current posterior belief
        root_q_values -- MF Q values at the current state
        policy_temp   -- inverse temperature
        policy_type   -- softmax / greeedy
        ----
        '''
        
        self.__dict__.update(**p)

        if self.max_seq_len is None:
            self.max_seq_len = self.horizon-1


        self._build_tree()
        self._build_qval_tree()

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

        if np.all(q_values == 0):
            return np.array([0.5, 0.5])

        if self.beta:
            t = self.beta
        else:
            t = 1
            
        if self.policy_type == 'softmax':
            return np.exp(q_values*t)/np.sum(np.exp(q_values*t))
        elif self.policy_type == 'greedy':
            if np.all(q_values == q_values.max()):
                a = np.random.choice([1, 0], p=[0.5, 0.5])
                return np.array([a, 1-a])
            else:
                return np.array(q_values >= q_values.max()).astype(int)
        else:
            raise KeyError('Unknown policy type')

    def evaluate_policy(self, qval_tree):

        '''
        ----
        Evaluate the tree policy

        qval_tree -- tree with Q values for each belief
        ----
        '''

        nqval_tree = deepcopy(qval_tree)
        eval_tree  = {hi:{} for hi in range(self.horizon)}

        # then propagate those values backwards
        for hi in reversed(range(self.horizon-1)):
            for idx, vals in self.belief_tree[hi].items():

                b  = vals[0]

                eval_tree[hi][idx] = 0

                qvals = nqval_tree[hi][idx].copy()
                probs = self._policy(qvals)

                next_idcs = vals[1]

                for next_idx in next_idcs:
                    a    = next_idx[0]
                    idx1 = next_idx[1]
                    idx2 = next_idx[2]

                    v_primes = [np.max(nqval_tree[hi+1][idx1]), np.max(nqval_tree[hi+1][idx2])]

                    b0 = b[a, 0]/np.sum(b[a, :])
                    b1 = b[a, 1]/np.sum(b[a, :])

                    eval_tree[hi][idx] += probs[a] * (b0*(1.0 + self.gamma*v_primes[0]) + b1*(0.0 + self.gamma*v_primes[1]))

        return eval_tree[0][0]

    def _belief_update(self, curr_belief, arm, rew):

        '''
        ----
        Bayesian belief updates for beta prior

        curr_belief -- matrix with the current beliefs
        arm         -- chosen arm 
        rew         -- received reward
        ----
        ''' 

        b_next = curr_belief.copy()
        if rew == 1:
            b_next[arm, 0] += 1
        else:
            b_next[arm, 1] += 1
        return b_next

    def _build_tree(self):

        '''
        ----
        Generate planning belief tree

        h -- horizon
        ----
        '''

        # initialise the hyperstate tree
        self.belief_tree = {hi:{} for hi in range(self.horizon)}
        
        idx = 0
        self.belief_tree[0][idx] = [self.root_belief, []]
    
        for hi in range(1, self.horizon):
            idx = 0
            for prev_idx, vals in self.belief_tree[hi-1].items():

                b = vals[0]

                for a in range(2):
                    
                    # success
                    r    = 1
                    b1s  = self._belief_update(b, a, r)
                    self.belief_tree[hi][idx] = [b1s, []]
                    
                    # fail
                    r    = 0
                    b1f  = self._belief_update(b, a, r)
                    self.belief_tree[hi][idx+1] = [b1f, []]

                    # add these to the previous belief
                    self.belief_tree[hi-1][prev_idx][-1] += [[a, idx, idx+1]]

                    idx += 2

        return None

    def full_updates(self):
        '''
        Compute full Bayes-optimal Q values at the root 
        (up to the specified horizon)
        ----
        tree  -- belief tree
        gamma -- discount factor
        ----
        '''

        self._build_qval_tree()

        # then propagate those values backwards
        for hi in reversed(range(self.horizon-1)):
            for idx, vals in self.belief_tree[hi].items():
                
                b  = vals[0]
                
                self.qval_tree[hi][idx] = np.zeros(2)

                next_idcs = vals[1]
                
                for next_idx in next_idcs:
                    a    = next_idx[0]
                    idx1 = next_idx[1]
                    idx2 = next_idx[2]

                    b0 = b[a, 0]/np.sum(b[a, :])
                    b1 = b[a, 1]/np.sum(b[a, :])
                    
                    v_primes = [np.max(self.qval_tree[hi+1][idx1]), np.max(self.qval_tree[hi+1][idx2])]
                    
                    self.qval_tree[hi][idx][a] = b0*(1.0 + self.gamma*v_primes[0]) + b1*(0.0 + self.gamma*v_primes[1])

        return np.max(self.qval_tree[0][0])

    def _build_qval_tree(self):
        
        self.qval_tree = {hi:{} for hi in range(self.horizon)}

        for hi in range(self.horizon):
            for idx, vals in self.belief_tree[hi].items():

                if self.init_qvals or (hi == (self.horizon - 1)):
                    b  = vals[0]
                    b0 = b[0, 0]/np.sum(b[0, :])
                    b1 = b[1, 0]/np.sum(b[1, :])
                    q_values = np.array([b0, b1])
                else:
                    q_values = np.zeros(2)
                
                self.qval_tree[hi][idx] = q_values.copy() # change temperature?

        return None

    def _build_need_tree(self):

        self.need_tree  = {hi:{} for hi in range(self.horizon)}

        self.need_tree[0][0] = 1
        for hi in range(1, self.horizon):
            for prev_idx, vals in self.belief_tree[hi-1].items():
                # compute Need with the default (softmax) policy
                prev_need    = self.need_tree[hi-1][prev_idx]
                policy_proba = self._policy(self.qval_tree[hi-1][prev_idx])
                
                b         = vals[0]
                b0 = b[0, 0]/np.sum(b[0, :])
                b1 = b[1, 0]/np.sum(b[1, :])
                
                next_idcs = vals[1]
                for next_idx in next_idcs:
                    a    = next_idx[0]
                    
                    idx1 = next_idx[1]
                    self.need_tree[hi][idx1] = policy_proba[a]*b0*prev_need*self.gamma

                    idx2 = next_idx[2]
                    self.need_tree[hi][idx2] = policy_proba[a]*b1*prev_need*self.gamma

        return None

    def _get_highest_evb(self, updates):
        
        max_evb = 0
        idx     = None
        for uidx, update in enumerate(updates):
            evb = update[-1][-1]
            if evb > max_evb:
                max_evb = evb
                idx     = uidx

        if max_evb > self.xi:
            return idx, max_evb
        else:
            return None, None

    def _generate_single_updates(self):

        updates = []

        for hi in reversed(range(self.horizon-1)):
                for idx, vals in self.belief_tree[hi].items():
                    
                    q    = self.qval_tree[hi][idx].copy() # current Q values of this belief state
                    b    = vals[0]
                    need = self.need_tree[hi][idx]

                    # compute the new (updated) Q value 
                    next_idcs = vals[1]
                    for next_idx in next_idcs:
                        
                        a    = next_idx[0]
                        idx1 = next_idx[1]
                        idx2 = next_idx[2]

                        v_primes = [np.max(self.qval_tree[hi+1][idx1]), np.max(self.qval_tree[hi+1][idx2])] # values of next belief states

                        # new (updated) Q value for action [a]
                        b0 = b[a, 0]/np.sum(b[a, :])
                        b1 = b[a, 1]/np.sum(b[a, :])

                        q_upd = b0*(1.0 + self.gamma*v_primes[0]) + b1*(0.0 + self.gamma*v_primes[1])

                        if a == 0:
                            q_new = np.array([q_upd, q[1]])
                        else:
                            q_new = np.array([q[0], q_upd])
                        
                        probs_before = self._policy(q)
                        probs_after  = self._policy(q_new)
                        gain         = np.dot(probs_after-probs_before, q_new)
                        evb          = need*gain
                        
                        if self.constrain_seqs:
                            if evb > self.xi:
                                updates += [[np.array([hi]), np.array([idx]), np.array([a]), q_new.reshape(1, -1).copy(), np.array([gain]), np.array([need]), np.array([evb])]]
                        else:
                            updates += [[np.array([hi]), np.array([idx]), np.array([a]), q_new.reshape(1, -1).copy(), np.array([gain]), np.array([need]), np.array([evb])]]
        return updates

    def _generate_forward_sequences(self, updates):

        seq_updates = []

        for update in updates:
        
            for l in range(self.max_seq_len - 1):

                if l == 0:
                    pool = [deepcopy(update)]
                else:
                    pool = deepcopy(tmp)

                tmp = []

                for seq in pool:

                    prev_hi  = seq[0][-1] # horizon of the previous update

                    if (prev_hi == self.horizon-2):
                        break 
                    
                    prev_idx = seq[1][-1] # idx of the previous belief
                    prev_a   = seq[2][-1] # previous action

                    # belief idcs from which we consider adding an action
                    prev_next_idcs = self.belief_tree[prev_hi][prev_idx][1]

                    for prev_next_idx in prev_next_idcs:
                        
                        if len(prev_next_idx) == 0:
                            break
                        
                        if prev_next_idx[0] == prev_a:
                            prev_idx1 = prev_next_idx[1]
                            prev_idx2 = prev_next_idx[2]

                            for idx in [prev_idx1, prev_idx2]:
                                
                                b = self.belief_tree[prev_hi+1][idx][0]
                                q = self.qval_tree[prev_hi+1][idx].copy()

                                next_idcs = self.belief_tree[prev_hi+1][idx][1]
                            
                                for next_idx in next_idcs:

                                    a    = next_idx[0]
                                    idx1 = next_idx[1]
                                    idx2 = next_idx[2]

                                    v_primes = [np.max(self.qval_tree[prev_hi+2][idx1]), np.max(self.qval_tree[prev_hi+2][idx2])] # values of next belief states

                                    # new (updated) Q value for action [a]
                                    b0 = b[a, 0]/np.sum(b[a, :])
                                    b1 = b[a, 1]/np.sum(b[a, :])

                                    q_upd = b0*(1.0 + self.gamma*v_primes[0]) + b1*(0.0 + self.gamma*v_primes[1])

                                    if a == 0:
                                        q_new = np.array([q_upd, q[1]])
                                    else:
                                        q_new = np.array([q[0], q_upd])
                                    
                                    probs_before = self._policy(q)
                                    probs_after  = self._policy(q_new)
                                    need         = self.need_tree[prev_hi+1][idx]
                                    gain         = np.dot(probs_after-probs_before, q_new)
                                    evb          = gain*need

                                    if self.constrain_seqs:
                                        if evb > self.xi:
                                            this_seq     = deepcopy(seq)
                                            this_seq[0]  = np.append(this_seq[0], prev_hi+1)
                                            this_seq[1]  = np.append(this_seq[1], idx)
                                            this_seq[2]  = np.append(this_seq[2], a)
                                            this_seq[3]  = np.vstack((this_seq[3], q_new.copy()))
                                            this_seq[4]  = np.append(this_seq[4], gain)
                                            this_seq[5]  = np.append(this_seq[5], need)
                                            this_seq[6]  = np.append(this_seq[6], np.dot(this_seq[4], this_seq[5]))
                                            tmp += [deepcopy(this_seq)]
                                    else:
                                        this_seq     = deepcopy(seq)
                                        this_seq[0]  = np.append(this_seq[0], prev_hi+1)
                                        this_seq[1]  = np.append(this_seq[1], idx)
                                        this_seq[2]  = np.append(this_seq[2], a)
                                        this_seq[3]  = np.vstack((this_seq[3], q_new.copy()))
                                        this_seq[4]  = np.append(this_seq[4], gain)
                                        this_seq[5]  = np.append(this_seq[5], need)
                                        this_seq[6]  = np.append(this_seq[6], np.dot(this_seq[4], this_seq[5]))
                                        tmp += [deepcopy(this_seq)]

                if len(tmp) > 0:
                    seq_updates += tmp

        return seq_updates

    def _generate_reverse_sequences(self, updates):

        seq_updates = []

        for update in updates:
        
            for l in range(self.max_seq_len - 1):

                if l == 0:
                    pool = [deepcopy(update)]
                else:
                    pool = deepcopy(tmp)

                tmp = []

                for seq in pool:

                    prev_hi  = seq[0][-1]

                    if (prev_hi == 0):
                        break 

                    prev_idx = seq[1][-1]
                    q_seq    = seq[3][-1, :].copy()

                    # find previous belief
                    for idx, vals in self.belief_tree[prev_hi-1].items():

                        next_idcs = vals[1]

                        for next_idx in next_idcs:

                            if (next_idx[1] == prev_idx) or (next_idx[2] == prev_idx):
                        
                                qval_tree = deepcopy(self.qval_tree)
                                qval_tree[prev_hi][prev_idx] = q_seq.copy()

                                q    = self.qval_tree[prev_hi-1][idx]
                                b    = vals[0]

                                a    = next_idx[0]
                                idx1 = next_idx[1]
                                idx2 = next_idx[2]

                                v_primes = [np.max(qval_tree[prev_hi][idx1]), np.max(qval_tree[prev_hi][idx2])] # values of next belief states

                                # new (updated) Q value for action [a]
                                b0 = b[a, 0]/np.sum(b[a, :])
                                b1 = b[a, 1]/np.sum(b[a, :])

                                q_upd = b0*(1.0 + self.gamma*v_primes[0]) + b1*(0.0 + self.gamma*v_primes[1])

                                if a == 0:
                                    q_new = np.array([q_upd, q[1]])
                                else:
                                    q_new = np.array([q[0], q_upd])
                                
                                probs_before = self._policy(q)
                                probs_after  = self._policy(q_new)
                                need         = self.need_tree[prev_hi-1][idx]
                                gain         = np.dot(probs_after-probs_before, q_new)
                                evb          = gain*need

                                if self.constrain_seqs:
                                    if evb > self.xi:
                                        this_seq     = deepcopy(seq)
                                        this_seq[0]  = np.append(this_seq[0], prev_hi-1)
                                        this_seq[1]  = np.append(this_seq[1], idx)
                                        this_seq[2]  = np.append(this_seq[2], a)
                                        this_seq[3]  = np.vstack((this_seq[3], q_new.copy()))
                                        this_seq[4]  = np.append(this_seq[4], gain)
                                        this_seq[5]  = np.append(this_seq[5], need)
                                        this_seq[6]  = np.append(this_seq[6], np.dot(this_seq[4], this_seq[5]))
                                        tmp += [deepcopy(this_seq)]
                                else:
                                    this_seq     = deepcopy(seq)
                                    this_seq[0]  = np.append(this_seq[0], prev_hi-1)
                                    this_seq[1]  = np.append(this_seq[1], idx)
                                    this_seq[2]  = np.append(this_seq[2], a)
                                    this_seq[3]  = np.vstack((this_seq[3], q_new.copy()))
                                    this_seq[4]  = np.append(this_seq[4], gain)
                                    this_seq[5]  = np.append(this_seq[5], need)
                                    this_seq[6]  = np.append(this_seq[6], np.dot(this_seq[4], this_seq[5]))
                                    tmp += [deepcopy(this_seq)]

                if len(tmp) > 0:
                    seq_updates += tmp

        return seq_updates

    def replay_updates(self):
        '''
        Perform replay updates in the belief tree
        '''

        self._build_qval_tree()
        self._build_need_tree()

        backups      = [None]
        qval_history = [deepcopy(self.qval_tree)]
        need_history = [deepcopy(self.need_tree)]

        # compute evb for every backup
        num = 1
        while True:

            updates = self._generate_single_updates()
            
            # generate sequences
            if self.sequences:
                
                fwd_seq_updates = self._generate_forward_sequences(updates)
                rev_seq_updates = self._generate_reverse_sequences(updates)

                if len(fwd_seq_updates) > 0:
                    updates += fwd_seq_updates
                if len(rev_seq_updates) > 0:
                    updates += rev_seq_updates

            uidx, evb = self._get_highest_evb(updates)

            if uidx is None:
                return qval_history, need_history, backups
            
            # execute update (replay) with the highest evb
            update = updates[uidx]
            his    = update[0]
            idcs   = update[1]
            aas    = update[2]
            q_news = update[3]
            evbs   = update[6]

            for idx, hi in enumerate(his):
                
                q_old = self.qval_tree[hi][idcs[idx]]
                q_new = q_news[idx, :].copy()
                self.qval_tree[hi][idcs[idx]] = q_new.copy()
                print('%u -- Replay %u/%u -- [%u, %u, %u], q_old: %.2f, q_new: %.2f, evb: %.3f'%(num, idx+1, len(his), hi, idcs[idx], aas[idx], q_old[aas[idx]], q_new[aas[idx]], evbs[idx]))

            # save history
            qval_history += [deepcopy(self.qval_tree)]
            need_history += [deepcopy(self.need_tree)]
            backups      += [[his, idcs, aas]]

            self._build_need_tree()
            num += 1