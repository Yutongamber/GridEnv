import numpy as np


class GeneralQ(object):

    def __init__(self, number_of_states, number_of_actions, initial_state,
                 target_policy, behaviour_policy, double, step_size=0.1):
        # Settings.
        self._number_of_actions = number_of_actions
        self._step_size = step_size
        self._behaviour_policy = behaviour_policy
        self._target_policy = target_policy
        self._double = double
        # Initial state.
        self._s = initial_state
        # Tabular q-estimates.
        self._q = np.zeros((number_of_states, number_of_actions))
        if double:
            self._q2 = np.zeros((number_of_states, number_of_actions))
        self._last_action = 0

    @property
    def q_values(self):
        return (self._q + self._q2) / 2 if self._double else self._q

    def step(self, reward, discount, next_state):
        next_action = self._behaviour_policy(self.q_values[next_state, :])

        if self._double is True:

            if np.random.random_sample() > 0.5:
                self._q[self._s, self._last_action] += self._step_size * (
                            reward + discount * self._q2[next_state, :] @ self._target_policy(self._q[next_state, :],
                                                                                              next_action) - self._q[
                                self._s, self._last_action])
            else:
                self._q2[self._s, self._last_action] += self._step_size * (
                            reward + discount * self._q[next_state, :] @ self._target_policy(self._q2[next_state, :],
                                                                                             next_action) - self._q2[
                                self._s, self._last_action])

        else:
            target_index = self._target_policy(self._q[next_state, :], next_action)
            target = reward + discount * (self._q[next_state, :] @ target_index)
            self._q[self._s, self._last_action] += self._step_size * (target - self._q[self._s, self._last_action])

        self._s = next_state

        self._last_action = next_action

        return next_action