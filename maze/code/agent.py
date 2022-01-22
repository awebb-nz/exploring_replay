import numpy as np
from copy import deepcopy

class Environment:

    def __init__(self, config, blocked_state_actions):

        '''
        ----
        config     -- matrix which specifies the env
        rew_matrix -- matrix which specifies reward received at each location
        ----
        '''

        self.config                = config
        self.blocked_state_actions = blocked_state_actions
        self.num_x_states          = config.shape[1]
        self.num_y_states          = config.shape[0]

        self.num_states            = self.num_x_states*self.num_y_states
        self.num_actions           = 4

        return None

    def _get_new_state(self, s, a, unlocked=False):

        '''
        ----
        s -- current state of the agent
        a -- chosen action
        ----
        '''

        y_coord, x_coord = self._convert_state_to_coords(s)

        # ----
        # first consider edge cases
        # at the top and choose up
        case1 = (y_coord == 0) and (a == 0)
        # at the bottom and choose down
        case2 = (y_coord == self.num_y_states - 1) and (a == 1)
        # at the left edge and choose left
        case3 = (x_coord == 0) and (a == 2)
        # at the right edge and choose right
        case4 = (x_coord == self.num_x_states - 1) and (a == 3)

        if case1 or case2 or case3 or case4:
            r = self.config[y_coord, x_coord]
            return s, r
        else:
            # ----
            # choose up
            if a == 0:
                x1_coord, y1_coord = x_coord, y_coord - 1
            # choose down
            elif a == 1:
                x1_coord, y1_coord = x_coord, y_coord + 1
            # choose left 
            elif a == 2:
                x1_coord, y1_coord = x_coord - 1, y_coord
            # choose right
            else:
                x1_coord, y1_coord = x_coord + 1, y_coord

            # check the barriers
            if ([s, a] not in self.blocked_state_actions) and (unlocked == False):
                r  = self.config[y1_coord, x1_coord]
                s1 = self._convert_coords_to_state([y1_coord, x1_coord])
                return s1, r
            else:
                r = self.config[y_coord, x_coord]
                return s, r
            

    def _convert_state_to_coords(self, s):

        y_coord = s // self.num_x_states
        x_coord = s % self.num_x_states

        return [y_coord, x_coord]

    def _convert_coords_to_state(self, coords: list):

        y_coord, x_coord = coords
        states = np.arange(self.num_states).reshape(self.num_y_states, self.num_x_states)

        return states[y_coord, x_coord]
        

class Agent(Environment):

    def __init__(self, config, start_coords, goal_coords, blocked_state_actions, uncertain_state_coords, uncertain_action, alpha, gamma, horizon, xi, policy_temp=None, policy_type='softmax'):

        super().__init__(config, blocked_state_actions)

        self.start_state = self._convert_coords_to_state(start_coords)
        self.goal_state  = self._convert_coords_to_state(goal_coords)

        self.Q = np.zeros((self.num_states, self.num_actions))
        # beta prior for the uncertain transition
        self.M = np.ones(2)
        # transition matrix for other transitions
        self.T = np.zeros((self.num_states, self.num_actions, self.num_states))
        for s in range(self.num_states):
            for a in range(self.num_actions):
                s1 = self._get_new_state(s, a)
                self.T[s, a, s1] = 1

        self.uncertain_state  = self._convert_coords_to_state(uncertain_state_coords)
        self.uncertain_action = uncertain_action

        self.policy_temp = policy_temp
        self.policy_type = policy_type
        self.alpha       = alpha
        self.gamma       = gamma
        self.horizon     = horizon
        self.xi          = xi

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

        if np.all(q_values == q_values.max()):
            return np.ones(self.num_actions)/self.num_actions

        if self.policy_temp:
            t = self.policy_temp
        else:
            t = 1
            
        if self.policy_type == 'softmax':
            return np.exp(q_values*t)/np.sum(np.exp(q_values*t))
        elif self.policy_type == 'greedy':
            if np.all(q_values == q_values.max()):
                a        = np.random.choice(range(self.num_actions), p=np.ones(self.num_actions)/self.num_actions)
                probs    = np.zeros(self.num_actions)
                probs[a] = 1
                return probs
            else:
                return np.array(q_values >= q_values.max()).astype(int)
        else:
            raise KeyError('Unknown policy type')

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

    def _qval_update(self, Q, s, a, r, s1):

        '''
        ----
        MF Q values update

        s           -- previous state 
        a           -- chosen action
        r           -- received reward
        s1          -- resulting new state
        ----
        ''' 

        qvals     = Q[s, :].copy()
        qvals[a] += self.alpha*(r + self.gamma*np.max(Q[s1, :]) - qvals[a])

        return qvals

    def _build_belief_tree(self, s):

        btree = {hi:{} for hi in range(self.horizon)}
        btree[0][(None, s, 0, 0)] = self.M

        for hi in range(1, self.horizon):
            c = 0
            for k, b in btree[hi-1].items():
                prev_c  = k[-1]
                prev_s1 = k[1]
                for a in range(self.num_actions):
                    if (prev_s1 == self.uncertain_state) and (a == self.uncertain_action):
                        s1 = self._get_new_state(prev_s1, a, unlocked=False)
                        b1 = self._belief_update(b, prev_s1, s1)
                        btree[hi][(a, s1, prev_c, c)] = b1
                        c += 1
                        s1 = self._get_new_state(prev_s1, a, unlocked=True)
                        b1 = self._belief_update(b, prev_s1, s1)
                        btree[hi][(a, s1, prev_c, c)] = b1
                        c += 1
                    else:
                        s1 = self._get_new_state(prev_s1, a)
                        btree[hi][(a, s1, prev_c, c)] = b
                        c += 1

        return btree

    def _build_qval_tree(self, btree):

        qtree = {hi:{} for hi in range(self.horizon)}

        for hi in range(self.horizon):
            for k, _ in btree[hi].items():
                s = k[1]
                qtree[hi][k] = self.Q[s, :]

        return qtree

    def _build_need_tree(self, btree, ntree):

        ntree = {hi:{} for hi in range(self.horizon)}

    def _replay(self):

        belief_trees = []
        qval_trees   = []
        need_trees   = []
        for s in range(self.num_states):
            belief_tree   = self._build_belief_tree(s)
            belief_trees += [belief_tree]
            qval_tree     = self._build_qval_tree(belief_tree)
            qval_trees   += [qval_tree]
            need_tree     = self._build_need_tree(belief_tree, qval_tree)
            need_trees   += [need_tree]

        return None

    def run_simulation(self, num_steps=100):

        s = self.start_state

        for _ in range(num_steps):

            probs = self._policy(self.Q[s, :])
            a     = np.random.choice(range(self.num_actions), p=probs)
            s1, r = self._get_new_state(s, a)

            self.Q[s, :] = self._qval_update(self.Q, s, a, r, s1)
            
            # check if attempted the shortcut
            if (s == self.uncertain_state) and (a == self.uncertain_action):
                self.M = self._belief_update(self.M, s, s1)

            if s1 == self.goal_state:
                s = self.start_state
            else:
                s = s1

        return None
