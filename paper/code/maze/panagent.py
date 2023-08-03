import numpy as np

class PanAgent():

    def __init__(self, **p):

        self.__dict__.update(**p)

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

        self.Q[s, a] += self.alpha*(r + self.gamma*np.nanmax(self.Q[s1, :]) - self.Q[s, a])

        return None

    def _mf_forget(self):

        self.Q = (1-self.mf_forget)*self.Q

        return None

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

        if temp is not None:
            t = temp
        else:
            t = self.beta

        if t != 'greedy':
            probs = np.exp(q_vals_allowed*t)/np.sum(np.exp(q_vals_allowed*t))
            if len(nan_idcs) > 0:
                for nan_idx in nan_idcs:
                    probs = np.insert(probs, nan_idx, 0)
            return probs
        elif t == 'greedy':
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

    def _compute_gain(self, q_before, q_after, inv_temp=None):

        '''
        ---
        Compute gain associated with each replay update
        ---
        '''

        if inv_temp == None:
            inv_temp = self.beta
        
        probs_before = self._policy(q_before, temp=inv_temp)
        probs_after  = self._policy(q_after, temp=inv_temp)

        return np.nansum((probs_after-probs_before)*q_after)

    def _compute_need(self, T, Q, inv_temp=None):
        
        '''
        ---
        Compute need associated with each state
        ---
        '''

        if inv_temp == None:
            inv_temp = self.beta

        Ts = np.zeros((self.num_states, self.num_states))
        for s in range(self.num_states):
            probs = self._policy(Q[s, :], temp=inv_temp)
            for a in range(self.num_actions):
                Ts[s, :] += probs[a]*T[s, a, :]
        
        return np.linalg.inv(np.eye(self.num_states) - self.gamma*Ts)