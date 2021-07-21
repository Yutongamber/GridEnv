import matplotlib.pyplot as plt
import numpy as np


W = -100  # wall
G = 100  # goal
GRID_LAYOUT = np.array([
    [W, W, W, W, W, W, W, W, W, W, W, W],
    [W, W, 0, W, W, W, W, W, W, 0, W, W],
    [W, 0, 0, 0, 0, 0, 0, 0, 0, G, 0, W],
    [W, 0, 0, 0, W, W, W, W, 0, 0, 0, W],
    [W, 0, 0, 0, W, W, W, W, 0, 0, 0, W],
    [W, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, W],
    [W, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, W],
    [W, W, 0, 0, 0, 0, 0, 0, 0, 0, W, W],
    [W, W, W, W, W, W, W, W, W, W, W, W]
])


class Grid(object):

    def __init__(self, noisy=False):
        # -1: wall
        # 0: empty, episode continues
        # other: number indicates reward, episode will terminate
        self._layout = GRID_LAYOUT
        self._start_state = (2, 2)
        self._state = self._start_state
        self._number_of_states = np.prod(np.shape(self._layout))
        self._noisy = noisy

    @property
    def number_of_states(self):
        return self._number_of_states

    def get_obs(self):
        y, x = self._state
        return y * self._layout.shape[1] + x

    def obs_to_state(obs):
        x = obs % self._layout.shape[1]
        y = obs // self._layout.shape[1]
        s = np.copy(grid._layout)
        s[y, x] = 4
        return s

    def step(self, action):
        y, x = self._state

        if action == 0:  # up
            new_state = (y - 1, x)
        elif action == 1:  # right
            new_state = (y, x + 1)
        elif action == 2:  # down
            new_state = (y + 1, x)
        elif action == 3:  # left
            new_state = (y, x - 1)
        else:
            raise ValueError("Invalid action: {} is not 0, 1, 2, or 3.".format(action))

        new_y, new_x = new_state
        reward = self._layout[new_y, new_x]
        if self._layout[new_y, new_x] == W:  # wall
            discount = 0.9
            new_state = (y, x)
        elif self._layout[new_y, new_x] == 0:  # empty cell
            reward = -1.
            discount = 0.9
        else:  # a goal
            discount = 0.
            new_state = self._start_state

        if self._noisy:
            width = self._layout.shape[1]
            reward += 10 * np.random.normal(0, width - new_x + new_y)

        self._state = new_state
        return reward, discount, self.get_obs()

    def plot_grid(self):
        plt.figure(figsize=(4, 4))
        plt.imshow(self._layout != W, interpolation="nearest", cmap='pink')
        plt.gca().grid(0)
        plt.xticks([])
        plt.yticks([])
        plt.title("The grid")
        plt.text(2, 2, r"$\mathbf{S}$", ha='center', va='center')
        plt.text(9, 2, r"$\mathbf{G}$", ha='center', va='center')
        h, w = self._layout.shape
        for y in range(h - 1):
            plt.plot([-0.5, w - 0.5], [y + 0.5, y + 0.5], '-k', lw=2)
        for x in range(w - 1):
            plt.plot([x + 0.5, x + 0.5], [-0.5, h - 0.5], '-k', lw=2)