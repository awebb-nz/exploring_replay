import numpy as np

class Environment:

    def __init__(self, config, blocked_state_actions: list, start_coords, goal_coords):

        '''
        ----
        config                -- matrix which specifies the env
        blocked_state_actions -- list with state-action pairs [s, a] which are blocked
        start_coords          -- start state coords
        goal_coords           -- goal state coords
        ----
        '''

        self.config                = config
        self.blocked_state_actions = blocked_state_actions
        self.num_x_states          = config.shape[1]
        self.num_y_states          = config.shape[0]

        self.num_states            = self.num_x_states*self.num_y_states
        self.num_actions           = 4

        self.start_coords          = start_coords
        self.goal_coords           = goal_coords

        return None

    def _get_new_state(self, s, a, unlocked=False):

        '''
        ----
        s        -- current state of the agent
        a        -- chosen action
        unlocked -- whether the action is available or not (for blocked_state_actions)
        ----
        '''

        if s == self.goal_state:
            return self.start_state, 0

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
            if (unlocked == True) or ([s, a] not in self.blocked_state_actions):
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