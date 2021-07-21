import numpy as np


class RandomTD(object):

    def __init__(self, number_of_states, number_of_actions, initial_state, step_size=0.1):
        self._values = np.zeros(number_of_states)
        self._state = initial_state
        self._number_of_actions = number_of_actions
        self._step_size = step_size

    @property
    def state_values(self):
        return self._values

    def step(self, reward, discount, next_state):
        next_action = np.random.randint(self._number_of_actions)

        self._values[self._state] += self._step_size * (
                    reward + discount * self._values[next_state] - self._values[self._state])
        #     print(self._values.shape)

        self._state = next_state
        return next_action