import numpy as np
from pys.utils.er import ReplayMemory
from pys.utils.per import ProportionalPrioritizedMemory
from pys.utils.her import HindsightMemory

class BaseAgent:
    def __init__(self, env:object, cfg:dict):
        self.state_size = env.observation_space.shape[0]
        self.action_size= env.action_space.shape[0]
        self.action_min = env.action_space.low[0]
        self.action_max = env.action_space.high[0]
        self.env_name   = cfg["ENV"]
        self.rl_type    = cfg["RL"]["ALGORITHM"]
        self.er_type    = cfg["ER"].upper()
        self.filename   = cfg["ENV"] + '_' + cfg["RL"]["ALGORITHM"] + '_' + cfg["ER"]

        # Experience Replay
        self.batch_size = cfg["BATCH_SIZE"]
        self.train_start = cfg["TRAIN_START"]
        self.buffer_size = cfg["MEMORY_SIZE"]
        if self.er_type == "ER":
            self.memory = ReplayMemory(capacity=self.buffer_size)
        elif self.er_type == "PER":
            self.memory = ProportionalPrioritizedMemory(capacity=self.buffer_size)
        elif self.er_type == "HER":
            self.memory = HindsightMemory(\
                capacity            = self.buffer_size,\
                replay_n            = cfg["HER"]["REPLAY_N"],\
                replay_strategy     = cfg["HER"]["STRATEGY"],\
                reward_func         = cfg["HER"]["REWARD_FUNC"],\
                done_func           = cfg["HER"]["DONE_FUNC"])
            self.filename = cfg["ENV"] + '_' + cfg["RL"]["ALGORITHM"] + '_' + cfg["ER"] + '_' + cfg["HER"]["STRATEGY"]

    def get_action(self,state):
        NotImplementedError

    def remember(self, state, action, reward, next_state, done, goal=None):
        state       = np.array(state,       dtype=np.float32)
        action      = np.array(action,      dtype=np.float32)
        reward      = np.array([reward],    dtype=np.float32)
        done        = np.array([done],      dtype=np.float32)
        next_state  = np.array(next_state,  dtype=np.float32)
        if self.er_type == "HER":
            goal        = np.array(goal,        dtype=np.float32)
            transition  = (state, action, reward, next_state, done, goal)
        else:
            transition  = (state, action, reward, next_state, done)
        self.memory.append(transition)
        return

    def train_model(self):
        NotImplementedError

    def load_model(self,at):
        '''
        Load pre-trained model
        '''
        NotImplementedError

    def save_model(self,at):
        '''
        Save trained model
        '''
        NotImplementedError
 
    def hard_update_target_model(self):
        NotImplementedError

    def soft_update_target_model(self):
        NotImplementedError