from environment import Environment
import numpy as np

        
class BamcpNeed(Environment):

    def __init__(self, config, start_coords, goal_coords, blocked_state_actions, uncertain_states_actions, gamma, Q, M, policy_temp=None, policy_type='softmax'):
        
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

        self.gamma       = gamma
        self.policy_temp = policy_temp
        self.policy_type = policy_type
        self.Q           = Q
        self.M           = M
        self.state       = self.start_state

        # initialise MF Q values
        self.T      = np.zeros((self.num_states, self.num_actions, self.num_states))

        # set edge Q values to np.nan
        for s in np.delete(range(self.num_states), self.goal_state):
            for a in range(self.num_actions):
                if (s == self.uncertain_states_actions[0]):
                    if (a == self.uncertain_states_actions[1]):
                        s1, _ = self._get_new_state(s, a, unlocked=True)
                        self.T[s, a, s1] = self.M[0]/(np.sum(self.M))
                        self.T[s, a, s]  = self.M[1]/(np.sum(self.M))
                else:
                    s1, _ = self._get_new_state(s, a, unlocked=False)
                    self.T[s, a, s1] = 1

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

    def search(self, s):
        
        self.eps  = 1e-6
        # {depth: [[history, n, T]]
        self.tree = {}

        self.simulate(s)
        return None

    def simulate(self, this_s):
        
        Ta = np.zeros((self.num_states, self.num_actions, self.num_states))

        # set edge Q values to np.nan
        for s in np.delete(range(self.num_states), self.goal_state):
            for a in range(self.num_actions):
                if (s == self.uncertain_states_actions[0]):
                    if (a == self.uncertain_states_actions[1]):
                        s1, _ = self._get_new_state(s, a, unlocked=True)
                        Ta[s, a, s1] = self.M[0]/(np.sum(self.M))
                        Ta[s, a, s]  = self.M[1]/(np.sum(self.M))
                else:
                    s1, _ = self._get_new_state(s, a, unlocked=False)
                    Ta[s, a, s1] = 1

        for a in range(self.num_actions):
            Ta[self.goal_state, a, self.start_state] = 1

        T = np.zeros((self.num_states, self.num_states))
        for s in range(self.num_states):
            qs    = self.Q[s, :]
            probs = self._policy(qs.copy()) 
            for a in range(self.num_actions):
                T[s, :] += probs[a] * Ta[s, a, :]

        self.SR = np.zeros((self.num_states, self.num_states))
        
        num_sims = 1500

        for sim in range(num_sims):
            if sim%100 == 0:
                print(sim)
            d  = 0
            Ts = np.zeros((self.num_states, self.num_states))
            m  = self.M.copy()
            s  = this_s
            while (self.gamma**d) > self.eps:

                new_m = m.copy()

                probs = self._policy(self.Q[s, :].copy())
                a     = np.random.choice(range(self.num_actions), p=probs)

                if (s == self.uncertain_states_actions[0]) and (a==self.uncertain_states_actions[1]):
                    s1u, _  = self._get_new_state(s, a, unlocked=True)
                    s1      = np.random.choice([s1u, s], p=[m[0]/np.sum(m), m[1]/np.sum(m)])
                    if (s1 == s1u):
                        new_m[0] += 1
                    else:
                        new_m[1] += 1
                else:
                    s1, _   = self._get_new_state(s, a)

                if s == self.goal_state:
                    s1 = self.start_state
                
                Ta = np.zeros((self.num_states, self.num_actions, self.num_states))

                # set edge Q values to np.nan
                for sx in np.delete(range(self.num_states), self.goal_state):
                    for a in range(self.num_actions):
                        if (sx == self.uncertain_states_actions[0]):
                            if (a == self.uncertain_states_actions[1]):
                                s1u, _ = self._get_new_state(sx, a, unlocked=True)
                                Ta[sx, a, s1u] = new_m[0]/(np.sum(new_m))
                                Ta[sx, a, sx]  = new_m[1]/(np.sum(new_m))
                        else:
                            s1l, _ = self._get_new_state(sx, a, unlocked=False)
                            Ta[sx, a, s1l] = 1

                for a in range(self.num_actions):
                    Ta[self.goal_state, a, self.start_state] = 1

                T = np.zeros((self.num_states, self.num_states))
                for sx in range(self.num_states):
                    qs    = self.Q[sx, :]
                    probs = self._policy(qs.copy()) 
                    for a in range(self.num_actions):
                        T[sx, :] += probs[a] * Ta[sx, a, :]
                
                Ts += (self.gamma**d)*np.linalg.matrix_power(T, d)

                m  = new_m.copy()
                s  = s1
                d += 1

            self.SR += Ts/num_sims

        return None