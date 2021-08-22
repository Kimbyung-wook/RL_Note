import random
import numpy as np

class HindsightMemory():
    def __init__(self, capacity:int, replay_strategy:str = 'random', replay_n:int = 8, reward_func = None, done_func = None):
        # Basic member
        self.buffer = []
        self.buffer_idx = 0
        self.capacity = capacity
        # HER members
        self.replay_n = replay_n
        self.replay_strategy = replay_strategy.lower()
        self.episode_buffer = []
        self.episode_buffer_idx = 0
        self.reward_func = reward_func
        self.done_func = done_func

    def append(self, sample:list) -> None:
        '''
        >>> HOW TO USE
        transition = (state, action, reward, next_state, done, goal)
        ReplayMemory.append(transition)
        '''
        self.add_transition(sample)
        self.add_her_transition(sample)
        done = bool(sample[4])
        if done:
            # print('Episode end           : buffer size : ', len(self.buffer), '/ and a epi steps : ', len(self.episode_buffer))
            self.generate_HER_transition()
            # print('after HER operation   : buffer size : ', len(self.buffer), '/ and a epi steps : ', len(self.episode_buffer))
            self.reset_current_episode()
            # print('reset current episode : buffer size : ', len(self.buffer), '/ and a epi steps : ', len(self.episode_buffer))

    def add_transition(self, sample:list) -> None:
        self.buffer_idx = self.buffer_idx % self.capacity
        if(len(self.buffer) < self.capacity):
            self.buffer += [sample]
        else:
            self.buffer[self.buffer_idx] = sample
        self.buffer_idx += 1

    # get samples from priority memory according mini batch size n
    def sample(self, n:int) -> list:
        '''
        >>> HOW TO USE
        mini_batch = ReplayMemory.sample(number_of_samples)

        # Sampling from the memory
        states      = np.array([sample[0] for sample in mini_batch])
        actions     = np.array([sample[1] for sample in mini_batch])
        rewards     = np.array([sample[2] for sample in mini_batch])
        next_states = np.array([sample[3] for sample in mini_batch])
        dones       = np.array([sample[4] for sample in mini_batch])
        goals       = np.array([sample[5] for sample in mini_batch])
        '''
        return random.sample(self.buffer,n)

    def __len__(self):
        return len(self.buffer)

    '''
    HER Operation
    '''

    def add_her_transition(self, sample:list) -> None:
        '''
            append a transition to current episode buffer
        '''
        self.episode_buffer += [sample]
        self.episode_buffer_idx += 1

    def reset_current_episode(self) -> None:
        '''
            reset current episode buffer
        '''
        self.episode_buffer = []
        self.episode_buffer_idx = 0

    def _sample_additional_goal(self):
        '''
            sample additional goal from current episode buffer
        '''
        if self.replay_strategy == 'final': # Get final state
            additional_goals= np.array([self.episode_buffer[-1][0]],dtype=np.float32)

        elif self.replay_strategy == 'future': 
            # ???? I dont understand
            mini_batch      = self.episode_buffer[-1]
            additional_goals= np.array([sample[0] for sample in mini_batch])

        elif self.replay_strategy == 'episode': # get states in current episode
            mini_batch      = random.sample(self.episode_buffer,self.replay_n)
            additional_goals= np.array([sample[0] for sample in mini_batch])

        elif self.replay_strategy == 'random': # get states in buffer
            mini_batch      = random.sample(self.buffer,self.replay_n)
            additional_goals= np.array([sample[0] for sample in mini_batch])

        # print('additional_goals ',additional_goals)
        return additional_goals

    def generate_HER_transition(self) -> None:
        # Replay current episode
        for episode in self.episode_buffer:
            state       = episode[0]
            action      = episode[1]
            reward      = episode[2]
            next_state  = episode[3]
            done        = episode[4]
            # Sample a set of additional goals for replay G := S (current episode)
            additional_goals = self._sample_additional_goal()
            # print('additional_goals ',additional_goals)
            # print('additional_goals ',len(additional_goals))
            for i in range(len(additional_goals)):
                now_goal = additional_goals[i]
                # Do achieve?
                do_achieve = np.array([self.done_func(state, action, reward, next_state, done, now_goal)],dtype=np.float32)
                new_reward = np.array([self.reward_func(state, action, reward, next_state, done, now_goal)],dtype=np.float32)
                transition = (state, action, new_reward, next_state, do_achieve, now_goal)
                # Store the transition to Buffer
                self.add_transition(transition)