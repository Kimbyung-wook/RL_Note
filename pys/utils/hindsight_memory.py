import numpy as np
import random
from .memory import ReplayMemory
from collections import deque

class HindsightMemory(ReplayMemory):
    def __init__(self, capacity, replay_strategy, replay_k, reward_func=None):
        self.capacity = capacity
        self.replay_strategy = replay_strategy
        self.replay_k = replay_k

        self.reward_func = reward_func
    
    def append(self, transition):
        '''
        >>> HOW TO USE
        transition = (state, action, reward, next_state, done, goal, instace_goal)
        ReplayMemory.append(transition)
        '''

        tmp = 0
        # for saving a episode
        done = transition[0][4] # get episode status, is it done?
        if(done == True): # Change episode index 
            tmp = 1
        else: # Keep episode index
            tmp = 1

        # Stack a transition
        self.buffer_idx = self.buffer_idx % self.capacity
        if(len(self.buffer) < self.capacity):
            self.buffer += [transition]
        else:
            self.buffer[self.buffer_idx] = transition
        self.buffer_idx += 1

    def sample(self, episode_batch, batch_size):

        if(self.replay_strategy == 'final'):
            transitions = (1,2,3)
        elif(self.replay_strategy == 'future'):
            transitions = (1,2,3)
        elif(self.replay_strategy == 'random'):
            transitions = random.sample(self.buffer,batch_size)
        elif(self.replay_strategy == 'episode'):
            transitions = (1,2,3)
        else: # Just Experience Replaying
            transitions = random.sample(self.buffer,batch_size)


        return transitions