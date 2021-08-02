import numpy as np
import random
import memory

class HindsightMemory(memory):
    def __init__(self, capacity, replay_strategy, replay_k, reward_func=None):
        self.capacity = capacity
        self.replay_strategy = replay_strategy
        self.replay_k = replay_k

        self.reward_func = reward_func
    
    def append(self, sample):
        '''
        >>> HOW TO USE
        transition = (state, action, reward, next_state, done, goal, instace_goal)
        ReplayMemory.append(transition)
        '''

        tmp = 0
        # for saving a episode
        done = sample[0][4] # get episode status, is it done?
        if(done == True): # Change episode index 
            tmp = 1
        else: # Keep episode index
            tmp = 1
            


    def sample(self, episode_batch, batch_size):

        if(self.replay_strategy == 'final'):
            transitions = (1,2,3)
        elif(self.replay_strategy == 'future'):
            transitions = (1,2,3)
        elif(self.replay_strategy == 'random'):
            transitions = (1,2,3)
        elif(self.replay_strategy == 'episode'):
            transitions = (1,2,3)
        else: # Just Experience Replaying
            transitions = random.sample(self.buffer,batch_size)


        return transitions