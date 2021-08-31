import numpy as np
import random

class ReplayMemory():
    def __init__(self, capacity):
        # Basic member
        self.buffer = []
        self.buffer_idx = 0
        self.capacity = capacity

    def append(self, transition:list):
        '''
        >>> HOW TO USE
        transition = (state, action, reward, next_state, done)
        ReplayMemory.append(transition)
        '''
        self.buffer_idx = self.buffer_idx % self.capacity
        if(len(self.buffer) < self.capacity):
            self.buffer += [transition]
        else:
            self.buffer[self.buffer_idx] = transition
        self.buffer_idx += 1

    def sample(self, n:int):
        '''
        >>> HOW TO USE
        mini_batch = ReplayMemory.sample(number_of_samples)

        # Sampling from the memory
        states      = np.array([sample[0] for sample in mini_batch])
        actions     = np.array([sample[1] for sample in mini_batch])
        rewards     = np.array([sample[2] for sample in mini_batch])
        next_states = np.array([sample[3] for sample in mini_batch])
        dones       = np.array([sample[4] for sample in mini_batch])
        '''
        return random.sample(self.buffer,n)

    def __len__(self):
        return len(self.buffer)

    def show(self):
        print(self.buffer)