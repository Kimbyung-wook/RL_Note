# it's from
# https://github.com/rlcode/per
import random
import numpy as np
from utils.sum_tree import SumTree

class PrioritizedMemory:  # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001
    epsilon = 0.000001  # Proportional Prioritization
    abs_err_upper = 1.000

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a

    def add(self, sample):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        # p = self._get_priority(error)
        self.tree.add(max_p, sample)

    def sample(self, n:int):
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

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
        # Rank-based Prioritization
        self.tree.update(idx, p)
        
    def batch_update(self, idxs, errors):
        for i in range(len(idxs)):
            p = self._get_priority(errors[i])
            self.tree.update(idxs[i], p)

    def __len__(self):
        return self.tree.n_entries
