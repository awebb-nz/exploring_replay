from environment import Environment
import numpy as np
import os

class Bamcp(Environment):

    def __init__(self, **p):
        
        '''
        ----
        Initialise the agent class
        ----
        '''
        
        self.__dict__.update(**p)

        # initialise the environment
        super().__init__(self.config, self.blocked_state_actions, self.start_coords, self.goal_coords)
        self.start_state = self._convert_coords_to_state(self.start_coords)
        self.goal_state  = self._convert_coords_to_state(self.goal_coords)
        
        # initialise MF Q values
        self.num_states  = self.config.shape[0]*self.config.shape[1]
        self.Q_nans      = np.zeros((self.num_states, self.num_actions))

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
                        self.Q_nans[s, a] = np.nan

        self.Q_ro = self.Q_nans.copy()

        return None

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

    def ucb_poliy(self, q_vals, c, n):

        nan_idcs = np.argwhere(np.isnan(q_vals)).flatten()
        if len(nan_idcs) > 0:
            q_vals_allowed = np.delete(q_vals, nan_idcs)
            n              = np.delete(n, nan_idcs)
        else:
            q_vals_allowed = q_vals

        for idx, q in enumerate(q_vals_allowed):
            if n[idx] == 0:
                q_vals_allowed[idx] = np.inf
            else:
                q_vals_allowed[idx] = q + c*np.sqrt(np.log(np.nansum(n))/n[idx])

        probs = np.zeros(len(q_vals_allowed))
        probs[np.nanargmax(q_vals_allowed)] = 1

        if len(nan_idcs) > 0:
            for nan_idx in nan_idcs:
                probs = np.insert(probs, nan_idx, 0)

        return probs

    def rollout_policy(self, q_vals):

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

        probs = np.ones(len(q_vals_allowed))/len(q_vals_allowed)
        if len(nan_idcs) > 0:
            for nan_idx in nan_idcs:
                probs = np.insert(probs, nan_idx, 0)
        return probs

    def search(self, s):
        
        self.Q    = self.Q_nans.copy()
        self.N    = self.Q_nans.copy()
        # {depth: [[history, q, N(s, a)]]
        self.tree = {0:[]}

        for i in range(self.num_sims):
            # if i%1000 == 0:
                # print(i)
            m = np.random.beta(self.M[0], self.M[1])
            _ = self.simulate([s], m, 0)

        q_vals = self.tree[0][0][1]#
        print(s, q_vals)
        return np.nanargmax(q_vals)

    def simulate(self, h, m, d):
        
        y, x = self.goal_coords
        rew  = self.config[y, x]
        if (self.gamma**d) * rew < self.eps:
            return 0

        s = h[-1]
        
        # if (s == self.goal_state):
        #     return 0

        check = False

        if d not in self.tree.keys():
            self.tree[d] = []

        lis   = self.tree[d]
        for vidx, v in enumerate(lis):
            if len(v) == 0:
                break
            if (h == v[0]):
                check = True
                break

        if not check:
            probs = self.rollout_policy(self.Q_ro[s, :].copy())
            if np.random.uniform() > 0.3:
                a = np.argwhere(self.Q_ro[s, :] == np.nanmax(self.Q_ro[s, :])).flatten()[0]
            else:
                a = np.random.choice(range(self.num_actions), p=probs) # can take all actions?

            if (s == self.uncertain_states_actions[0]) and (a==self.uncertain_states_actions[1]):
                s1u, _  = self._get_new_state(s, a, unlocked=True)
                s1      = np.random.choice([s1u, s], p=[m, 1-m])
                y, x    = self._convert_state_to_coords(s1)
                r       = self.config[y, x]
            else:
                s1, r   = self._get_new_state(s, a)

            if s == self.goal_state:
                s1 = self.start_state
            
            h1 = h + [a, s1]
            R  = r + self.gamma * self.rollout(h1, m, d)

            q     = self.Q[s, :].copy()
            q[a]  = R
            n     = self.N[s, :].copy()
            n[a] += 1
            self.tree[d] += [[h, q.copy(), n]]
            # print(d, s, R)
            return R

        q      = v[1]
        n      = v[2]
        probs  = self.ucb_poliy(q.copy(), self.c, n)
        a      = np.argwhere(probs == 1).flatten()[0]

        if (s == self.uncertain_states_actions[0]) and (a==self.uncertain_states_actions[1]):
                s1u, _  = self._get_new_state(s, a, unlocked=True)
                s1      = np.random.choice([s1u, s], p=[m, 1-m])
                y, x    = self._convert_state_to_coords(s1)
                r       = self.config[y, x]
        else:
            s1, r   = self._get_new_state(s, a)

        if s == self.goal_state:
            s1 = self.start_state

        h1 = h + [a, s1]
        R  = r + self.gamma * self.simulate(h1, m, d+1)

        n[a] += 1
        q[a] += (R-q[a])/n[a]
        self.tree[d][vidx] = [h, q.copy(), n]

        return R

    def rollout(self, h, m, d):

        y, x = self.goal_coords
        rew  = self.config[y, x]
        if (self.gamma**d) * rew < self.eps:
            # print('nope', h)
            return 0

        s = h[-1]
        
        # if (s == self.goal_state):
        #     # print('yes', h)
        #     return 0

        probs = self.rollout_policy(self.Q_ro[s, :].copy())
        if np.random.uniform() > 0.3:
            a = np.argwhere(self.Q_ro[s, :] == np.nanmax(self.Q_ro[s, :])).flatten()[0]
        else:
            a = np.random.choice(range(self.num_actions), p=probs) # can take all actions?

        if (s == self.uncertain_states_actions[0]) and (a==self.uncertain_states_actions[1]):
            s1u, _  = self._get_new_state(s, a, unlocked=True)
            s1      = np.random.choice([s1u, s], p=[m, 1-m])
            y, x    = self._convert_state_to_coords(s1)
            r       = self.config[y, x]
        else:
            s1, r   = self._get_new_state(s, a)
        
        if s == self.goal_state:
            s1 = self.start_state

        h1 = h + [a, s1]
        return r + self.gamma * self.rollout(h1, m, d+1)

    def run_simulation(self, save_path=None):

        if save_path is not None:
            f = open(os.path.join(save_path, 'info.txt'), 'w')

        self.state = self.start_state

        for ep in range(self.num_steps):

            a     = self.search(self.state)
            s1, r = self._get_new_state(self.state, a, unlocked=False)
            
            # update transition probability belief
            if (self.state == self.uncertain_states_actions[0]) and (a==self.uncertain_states_actions[1]):
                self.M = self._belief_update(self.M, self.state, s1)

            self.Q_ro[self.state, a] += self.alpha*(r + self.gamma*np.nanmax(self.Q_ro[s1, :]) - self.Q_ro[self.state, a])

            if save_path is not None:
                f.write('\nMove %u/%u, [%u, %u, %u, %u]'%(ep+1, self.num_steps, self.state, a, r, s1))

            self.state = s1

            if self.state == self.goal_state:
                self.state = self.start_state

        f.close()
        return None