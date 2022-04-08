from environment import Environment
import numpy as np

class Bamcp(Environment):

    def __init__(self, config, start_coords, goal_coords, blocked_state_actions, uncertain_states_actions, gamma):
        
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
        ----
        '''
        
        super().__init__(config, blocked_state_actions, start_coords, goal_coords)

        self.start_state = self._convert_coords_to_state(start_coords)
        self.goal_state  = self._convert_coords_to_state(goal_coords)

        self.uncertain_states_actions = uncertain_states_actions

        self.gamma       = gamma

        self.state       = self.start_state

        # initialise MF Q values
        self.Q_nans = np.zeros((self.num_states, self.num_actions))

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

        # beta prior for the uncertain transition
        self.M = np.ones(2)

        return None

    def ucb_poliy(self, q_vals, c, n):

        nan_idcs = np.argwhere(np.isnan(q_vals)).flatten()
        if len(nan_idcs) > 0:
            q_vals_allowed = np.delete(q_vals, nan_idcs)
            n              = np.delete(n, nan_idcs)
        else:
            q_vals_allowed = q_vals

        for idx, q in enumerate(q_vals_allowed):
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
        
        self.c    = 1
        self.M    = np.array([1, 1])
        self.eps  = 1e-7
        self.Q    = self.Q_nans.copy()
        self.N    = self.Q_nans.copy() + 1
        # {depth: [[history, q, N(s, a)]]
        self.tree = {0:[]}

        for i in range(2000):
            # if i%1000 == 0:
                # print(i)
            m = np.random.beta(self.M[0], self.M[1])
            _ = self.simulate([s], m, 0)

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
            probs = self.rollout_policy(self.Q[s, :].copy())
            if np.random.uniform() > 0.5:
                a = np.argwhere(self.Q[s, :] == np.nanmax(self.Q[s, :])).flatten()[0]
            else:
                a = np.random.choice(range(self.num_actions), p=probs)

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

        q[a] += (R-q[a])/n[a]
        n[a] += 1
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

        probs = self.rollout_policy(self.Q[s, :].copy())
        if np.random.uniform() > 0.5:
            a = np.argwhere(self.Q[s, :] == np.nanmax(self.Q[s, :])).flatten()[0]
        else:
            a = np.random.choice(range(self.num_actions), p=probs)

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