# it's from
# https://github.com/rlcode/per
# Find RL_Note path and append sys path

import random
import numpy as np
from .sum_tree import SumTree

class ProportionalPrioritizedMemory:  # stored as ( s, a, r, s_ ) in SumTree
    alpha = 0.6     # For proportional variant
    beta = 0.4
    # a = 0.7     # For rank-based variant
    # beta = 0.5
    alpha_increment_per_sampling = 0.001
    beta_increment_per_sampling = 0.001
    epsilon = 0.0001  # Proportional Prioritization
    abs_err_upper = 1.000

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity

    def _get_priority(self, error):
        '''
        It is NOT Upper P(i)
        '''
        return (np.abs(error) + self.epsilon) ** self.alpha

    def append(self, sample:list):
        '''
        >>> HOW TO USE
        transition = (state, action, reward, next_state, done, td_error)
        ReplayMemory.append(transition)
        '''
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p <= 0.001:
            max_p = self.abs_err_upper
        # max_p = self._get_priority(error)
        self.tree.add(max_p, sample)

    def sample(self, n:int):
        batch, idxs, priorities = [], [], []
        segment = self.tree.total() / n

        self.alpha = np.max([0., self.alpha - self.alpha_increment_per_sampling])
        self.beta  = np.min([1., self.beta  + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight

    def update(self, idx:int, error:float):
        # Proportional Prioritization
        p = self._get_priority(abs(error) + self.epsilon)
        self.tree.update(idx, p)

    def __len__(self):
        return self.tree.n_entries
