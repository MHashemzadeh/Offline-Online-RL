import numpy as np
import matplotlib.pyplot as plt

from utils.torch_utils import random_seed


class GridHardXY:
    def __init__(self, seed=np.random.randint(int(1e5))):
        self.seed(seed)
        self.state_dim = (2,)
        self.action_dim = 4
        self.obstacles_map = self.get_obstacles_map()
        self.actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        self.min_x, self.max_x, self.min_y, self.max_y = 0, 14, 0, 14
        self.goal_x, self.goal_y = 9, 9
        self.current_state = None
        self.transition_counter = 0
        self.timeout = 100

    def generate_state(self, coords):
        return np.array(coords)

    def info(self, key):
        return

    def reset(self):
        while True:
            rand_state = np.random.randint(low=0, high=15, size=2)
            rx, ry = rand_state
            if not int(self.obstacles_map[rx][ry]) and not (rx == self.goal_x and ry == self.goal_y):
                self.current_state = rand_state[0], rand_state[1]
                self.transition_counter = 0
                return self.generate_state(self.current_state)

    def step(self, a):

        dx, dy = self.actions[a]
        x, y = self.current_state

        nx = x + dx
        ny = y + dy

        nx, ny = min(max(nx, self.min_x), self.max_x), min(max(ny, self.min_y), self.max_y)

        if not self.obstacles_map[nx][ny]:
            x, y = nx, ny

        self.current_state = x, y
        self.transition_counter += 1 
        # print(self.transition_counter)
        if (x == self.goal_x and y == self.goal_y):
            return self.generate_state([x, y]), 1.0, True, ""
        elif self.transition_counter == self.timeout:
            return self.generate_state([x, y]), 0.0, True, ""
        else:
            return self.generate_state([x, y]), 0.0, False, ""

    def get_visualization_segment(self):
        state_coords = [[x, y] for x in range(15)
                       for y in range(15) if not int(self.obstacles_map[x][y])]
        states = [self.generate_state(coord) for coord in state_coords]
        # goal_coords = [[9, 9], [0, 0], [14, 0], [7, 14]]
        # goal_coords = [[9, 9], [9, 12], [13, 11], [8, 6], [13, 2]] # 0 5 25 50 100
        # goal_coords = [[9, 9], [13, 11], [8, 6], [9, 1], [13, 2], [4, 13], [1, 3]] # 0 25 50 75 100 125 150
        goal_coords = [[9, 9], [3, 4], [1, 0], [0, 14], [3, 14], [7, 14], [10, 3], [14, 4]] #
        goal_states = [self.generate_state(coord) for coord in goal_coords]
        return np.array(states), np.array(state_coords), np.array(goal_states), np.array(goal_coords)

    def get_obstacles_map(self):
        _map = np.zeros([15, 15])
        _map[2, 0:6] = 1.0
        _map[2, 8:] = 1.0
        _map[3, 5] = 1.0
        _map[4, 5] = 1.0
        _map[5, 2:7] = 1.0
        _map[5, 9:] = 1.0
        _map[8, 2] = 1.0
        _map[8, 5] = 1.0
        _map[8, 8:] = 1.0
        _map[9, 2] = 1.0
        _map[9, 5] = 1.0
        _map[9, 8] = 1.0
        _map[10, 2] = 1.0
        _map[10, 5] = 1.0
        _map[10, 8] = 1.0
        _map[11, 2:6] = 1.0
        _map[11, 8:12] = 1.0
        _map[12, 5] = 1.0
        _map[13, 5] = 1.0
        _map[14, 5] = 1.0

        return _map

    def get_useful(self, state=None):
        if state:
            return state
        else:
            return self.current_state

    def seed(self, seed):
       random_seed(seed) 


class GridHardGS(GridHardXY):
    def __init__(self, seed=np.random.randint(int(1e5))):
        super().__init__(seed)

        d = len(self.obstacles_map)
        self.state_dim = (d, d, 1)

        """
        # Gray-scale image
            walls are black: 0.0
            agent is gray:   128.0
            open space is white: 255.0
        """
        self.gray_template = np.ones(self.state_dim) * 255.0
        for x in range(d):
            for y in range(d):
                if self.obstacles_map[x][y]:
                    self.gray_template[x][y] = 0.0

    def generate_state(self, coords):
        state = np.copy(self.gray_template)
        x, y = coords
        state[x][y] = 128.0
        return state

    def get_features(self, state):
        raise NotImplementedError


class GridHardRGB(GridHardXY):
    def __init__(self, seed=np.random.randint(int(1e5))):
        super().__init__(seed)

        d = len(self.obstacles_map)
        self.state_dim = (d, d, 3)

        """
        # Gray-scale image
            Walls are Red
            Open spaces are Green
            Agent is Blue
        """
        self.rgb_template = np.zeros(self.state_dim)
        for x in range(d):
            for y in range(d):
                if self.obstacles_map[x][y]:
                    self.rgb_template[x][y][0] = 255.0
                else:
                    self.rgb_template[x][y][1] = 255.0

    def generate_state(self, coords):
        state = np.copy(self.rgb_template)
        x, y = coords
        assert state[x][y][1] == 255.0 and state[x][y][2] == 0.0

        state[x][y][1] = 0.0    # setting the green color on
        state[x][y][2] = 255.0  # turning the blue color on
        return np.moveaxis(state, -1, 0) #NOTE: Made it channel last

    def get_features(self, state):
        raise NotImplementedError

    def get_useful(self, state=None):
        blue = np.array([0., 0., 255.])
        if state is None:
            state = self.generate_state(self.current_state)
        idx = np.where(np.all(state==blue, axis=2) == True)
        coord = np.array([idx[0][0], idx[1][0]])
        return coord


class GridHardRGBGoalAll(GridHardRGB):
    def __init__(self, goal_id, seed=np.random.randint(int(1e5))):
        super().__init__(seed)
        # self.nos = (self.state_dim[0] * self.state_dim[1]) - int(np.sum(self.obstacles_map))
        self.goals = [[i, j] for i in range(self.state_dim[0]) \
                              for j in range(self.state_dim[1]) if not self.obstacles_map[i, j]]
        self.goal_x, self.goal_y = self.goals[goal_id]
        self.goal_state_idx = goal_id

    def get_goal(self):
        return self.goal_state_idx, [self.goal_x, self.goal_y]

    def get_goals_list(self):
        return self.goals

    def visualize_goal_id(self):
        ids = np.zeros((self.state_dim[0], self.state_dim[1]))
        for idx, xy in enumerate(self.goals):
            ids[xy[0], xy[1]] = idx
        plt.figure()
        plt.imshow(ids, interpolation='nearest', cmap="Blues")
        for k in range(self.state_dim[0]):
            for j in range(self.state_dim[1]):
                if ids[k, j] != 0:
                    plt.text(j, k, "{:1.0f}".format(ids[k, j]),
                             ha="center", va="center", color="orange")
        plt.show()