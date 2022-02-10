import numpy as np
from copy import deepcopy

class Tree:

    def __init__(self, root_belief, root_q_values, policy_temp, policy_type):
        
        '''
        ----
        MAB agent

        root_belief   -- current posterior belief
        root_q_values -- MF Q values at the current state
        policy_temp   -- inverse temperature
        policy_type   -- softmax / greeedy
        ----
        '''
        
        self.root_belief   = root_belief
        self.root_q_values = root_q_values
        self.policy_temp   = policy_temp
        self.policy_type   = policy_type

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

        if self.policy_temp:
            t = self.policy_temp
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

        eval_tree  = {hi:{} for hi in range(self.horizon)}

        # then propagate those values backwards
        for hi in reversed(range(self.horizon-1)):
            for k, b in self.belief_tree[hi].items():
                
                probs = self._policy(qval_tree[hi][k])
                
                eval_tree[hi][k] = 0

                c        = k[-1]
                for a in range(2):
                    v_primes = []
                    if hi == (self.horizon - 2):
                        for k1, q1 in qval_tree[hi+1].items():
                            prev_c = k1[-2]
                            prev_a = k1[0]
                            if (prev_c == c) and (prev_a == a):
                                v_primes += [np.dot(self._policy(q1), q1)]
                                if len(v_primes) == 2:
                                    break
                    else:
                        for k1, q1 in eval_tree[hi+1].items():
                            prev_c = k1[-2]
                            prev_a = k1[0]
                            if (prev_c == c) and (prev_a == a):
                                v_primes += [np.dot(self._policy(q1), q1)]
                                if len(v_primes) == 2:
                                    break
                    
                    b0 = b[a, 0]/np.sum(b[a, :])
                    b1 = b[a, 1]/np.sum(b[a, :])

                    eval_tree[hi][k] += probs[a]*(b0*(1.0 + self.gamma*v_primes[0]) + b1*(0.0 + self.gamma*v_primes[1]))

        return eval_tree[0]

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

    def build_tree(self, horizon):

        '''
        ----
        Generate planning belief tree

        h -- horizon
        ----
        '''

        self.horizon = horizon

        # initialise the hyperstate tree
        self.belief_tree = {hi:{} for hi in range(self.horizon)}
        
        self.belief_tree[0][(0, 0, 0)] = self.root_belief
    
        for hi in range(1, self.horizon):
            c = 0
            if hi == 1:
                for a in range(2):
                    for r in [1, 0]:
                        b1 = self._belief_update(self.root_belief, a, r)
                        self.belief_tree[hi][(a, 0, c)] = b1
                        c += 1
            else:
                for k, v in self.belief_tree[hi-1].items():
                    prev_c = k[-1]
                    for a in range(2):
                        for r in [1, 0]:
                            b1 = self._belief_update(v, a, r)
                            self.belief_tree[hi][(a, prev_c, c)] = b1
                            c += 1
        return None

    def full_updates(self, gamma):
        '''
        Compute full Bayes-optimal Q values at the root 
        (up to the specified horizon)
        ----
        tree  -- belief tree
        gamma -- discount factor
        ----
        '''

        self.gamma      = gamma
        self.qval_tree  = {hi:{} for hi in range(self.horizon)}

        # first asign values to leaf nodes -- immediate reward
        hi = self.horizon - 1
        for k, b in self.belief_tree[hi].items():
            b0 = b[0, 0]/np.sum(b[0, :])
            b1 = b[1, 0]/np.sum(b[1, :])
            self.qval_tree[hi][k] = np.array([b0, b1])

        # then propagate those values backwards
        for hi in reversed(range(self.horizon-1)):
            for k, b in self.belief_tree[hi].items():
                self.qval_tree[hi][k] = np.zeros(2)

                c        = k[-1]
                for a in range(2):
                    v_primes = []
                    for k1, q1 in self.qval_tree[hi+1].items():
                        prev_c = k1[-2]
                        prev_a = k1[0]
                        if (prev_c == c) and (prev_a == a):
                            v_primes += [np.max(q1)]
                            if len(v_primes) == 2:
                                break
                    
                    b0 = b[a, 0]/np.sum(b[a, :])
                    b1 = b[a, 1]/np.sum(b[a, :])
                    self.qval_tree[hi][k][a] = b0*(1.0 + self.gamma*v_primes[0]) + b1*(0.0 + self.gamma*v_primes[1])

        return None

    def replay_updates(self, gamma, xi):
        '''
        Perform replay updates in the belief tree
        ----
        tree  -- belief tree
        gamma -- discount factor
        xi    -- EVB threshold
        ----
        '''

        self.gamma = gamma
        self.xi    = xi

        self.evb_tree   = {hi:{} for hi in range(self.horizon)} # tree with evb values for each node
        self.qval_tree  = {hi:{} for hi in range(self.horizon)} # tree with Q value estimates for each node
        self.need_tree  = {hi:{} for hi in range(self.horizon)} # tree with Need estimates for each node
        backups         = [None] # list to save replay updates

        qval_history = [] # same here
        need_history = []

        # first assign initial q values & compute the initial Need
        for hi in range(self.horizon):
            for k, b in self.belief_tree[hi].items():
                
                # if hi == 0:
                    # self.qval_tree[hi][k] = self.root_q_values.copy()
                # else:
                b0 = b[0, 0]/np.sum(b[0, :])
                b1 = b[1, 0]/np.sum(b[1, :])
                q_values = np.array([b0, b1])
                
                self.qval_tree[hi][k] = q_values.copy() # change temperature?

                # compute Need with the default (softmax) policy
                prev_c = k[-2]
                c      = k[-1]
                a      = k[0]
                proba  = 1
                for hin in reversed(range(hi)):
                    for kn, bn in self.belief_tree[hin].items():
                        if kn[-1] == prev_c:
                            policy_proba = self._policy(self.qval_tree[hin][kn])
                            bc           = bn[a, c%2]/np.sum(bn[a, :])
                            proba       *= policy_proba[a]*bc

                            c      = kn[-1]
                            prev_c = kn[-2]
                            a      = kn[0]
                            break

                self.need_tree[hi][k] = proba

        qval_history += [deepcopy(self.qval_tree)]
        need_history += [deepcopy(self.need_tree)]

        # compute evb for every backup
        num = 1
        while True:
            max_evb = 0

            nqval_tree = {hi:{} for hi in range(self.horizon)} # tree with new (updated) Q value estimates for each node
            for hi in reversed(range(self.horizon-1)):
                for k, b in self.belief_tree[hi].items():
                    
                    q        = self.qval_tree[hi][k].copy() # current Q values of this belief state
                    v        = np.dot(self._policy(q), q)   # value of this belief state

                    # -- probability of reaching this belief state (Need) -- #
                    # again, computed with the default (softmax) policy
                    prev_c = k[-2]
                    c      = k[-1]
                    a      = k[0]
                    proba  = 1
                    for hin in reversed(range(hi)):
                        for kn, bn in self.belief_tree[hin].items():
                            if kn[-1] == prev_c:
                                policy_proba = self._policy(self.qval_tree[hin][kn])
                                bc           = bn[a, c%2]/np.sum(bn[a, :])
                                proba       *= policy_proba[a]*bc

                                c      = kn[-1]
                                prev_c = kn[-2]
                                a      = kn[0]
                                break

                    self.need_tree[hi][k] = proba
                            
                    v_primes = []

                    # compute the new (updated) Q value 
                    c = k[-1]
                    for a in range(2):
                        v_primes = []
                        for k1, q1 in self.qval_tree[hi+1].items():
                            prev_c = k1[-2]
                            if prev_c == c and k1[0] == a:
                                v_primes += [np.max(q1)] # values of next belief states
                                # v_primes += [np.dot(self._policy(q1), q1)]
                                if len(v_primes) == 2:
                                    break

                        # new (updated) Q value for action [a]
                        b0 = b[a, 0]/np.sum(b[a, :])
                        b1 = b[a, 1]/np.sum(b[a, :])
                        q_upd = b0*(1.0 + self.gamma*v_primes[0]) + b1*(0.0 + self.gamma*v_primes[1])


                        if a == 0:
                            q_new = np.array([q_upd, q[1]])
                        else:
                            q_new = np.array([q[0], q_upd])
                        
                        v_new   = np.dot(self._policy(q_new), q_new) 

                        new_key = tuple(list(k) + [a])
                        
                        evb   = proba*(v_new - v)
                        # evb = (v_new - v)
                            
                        if evb > max_evb:
                            max_evb = evb

                        self.evb_tree[hi][new_key]   = evb
                        nqval_tree[hi][new_key] = q_upd

            # break the loop based on \xi evb threshold
            if max_evb <= self.xi:
                break

            max_val = 0
            for hi in reversed(range(self.horizon-1)):
                for k, v in self.evb_tree[hi].items():
                    if v > max_val:
                        backup  = [hi, k]
                        max_val = v
            if max_val <= 0:
                return qval_history, need_history, backups

            # execute update (replay) with the highest evb
            hi = backup[0]
            k  = backup[1][:-1]
            a  = backup[1][-1]

            qvals    = self.qval_tree[hi][k]
            new_qval = nqval_tree[hi][backup[1]]
            qvals[a] = new_qval
            self.qval_tree[hi][k] = qvals

            # print('Replay %u'%num)
            num += 1

            # save history
            qval_history += [deepcopy(self.qval_tree)]
            need_history += [deepcopy(self.need_tree)]
            backups      += [[self.belief_tree[hi][k], backup[0], backup[1]]]

            print('\n')
            print(backup, max_val)
            print(self.evaluate_policy(self.qval_tree))

        return  qval_history, need_history, backups