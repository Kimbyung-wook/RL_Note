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
    states    = np.array([sample[0] for sample in mini_batch])
    actions   = np.array([sample[1] for sample in mini_batch])
    rewards   = np.array([sample[2] for sample in mini_batch])
    next_states = np.array([sample[3] for sample in mini_batch])
    dones     = np.array([sample[4] for sample in mini_batch])
    '''
    return random.sample(self.buffer,n)

  def __len__(self):
    return len(self.buffer)

  def show(self):
    print(self.buffer)

  def reset(self):
    self.buffer = []
    self.buffer_idx = 0

class ReplayMemory1():
  '''
  This is OpenAI baseline style-Replay Memory
  Refer from
  https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
  '''
  def __init__(self, capacity:int)->None:
    self._buffer = []
    self._buffer_idx = 0
    self._capacity = capacity
    return

  def append(self, obs, action, reward, next_obs, done)->None:
    transition = (obs, action, reward, next_obs, done)
    self.buffer_idx = self.buffer_idx % self._capacity
    if(len(self._buffer) < self._capacity):
      self._buffer.append(transition)
    else:
      self._buffer[self.buffer_idx] = transition
    self.buffer_idx += 1

  def append(self, transition:list)->None:
    self.buffer_idx = self.buffer_idx % self._capacity
    if(len(self._buffer) < self._capacity):
      self._buffer.append(transition)
    else:
      self._buffer[self.buffer_idx] = transition
    self.buffer_idx += 1

  def _encode_sample(self, idxes):
    obss, actions, rewards, next_obss, dones = [], [], [], [], []
    for i in idxes:
      sample = self._buffer[i]
      obs, action, reward, next_obs, done = sample
      actions.append(   np.array(obs, copy=False))
      obss.append(      np.array(action, copy=False))
      rewards.append(   np.array(reward, copy=False))
      next_obss.append( np.array(next_obs, copy=False))
      dones.append(     np.array(done, copy=False))
    return  np.array(obss), \
            np.array(actions),\
            np.array(rewards),\
            np.array(next_obss),\
            np.array(dones)

  def sample(self, n:int):
    idxes = [random.randint(0, len(self._buffer)-1) for _ in range(n)]
    return self._encode_sample(idxes=idxes)

  def reset(self):
    self._buffer = []
    self._buffer_idx = 0

class ReplayMemory_nparray5():
  def __init__(self, capacity:int)->None:
    self.items = {
      '0':None,
      '1':None,
      '2':None,
      '3':None,
      '4':None,
    }
    self.buffer_idx = 0
    self.capacity = capacity
    return 
  
  def append(self, transition:list)->None:
    item0 = np.array([transition[0]])
    item1 = np.array([transition[1]])
    item2 = np.array([transition[2]])
    item3 = np.array([transition[3]])
    item4 = np.array([transition[4]])
    if self.buffer_idx is 0: # First append
      self.items['0'] = item0
      self.items['1'] = item1
      self.items['2'] = item2
      self.items['3'] = item3
      self.items['4'] = item4
    elif self.buffer_idx < self.capacity: # Buffer is not full
      self.items['0'] = np.append(self.items['0'], item0, axis=0) 
      self.items['1'] = np.append(self.items['1'], item1, axis=0) 
      self.items['2'] = np.append(self.items['2'], item2, axis=0) 
      self.items['3'] = np.append(self.items['3'], item3, axis=0) 
      self.items['4'] = np.append(self.items['4'], item4, axis=0) 
    else: # Buffer is full
      idx = self.buffer_idx % self.capacity
      self.items['0'][self.buffer_idx] = item0
      self.items['1'][self.buffer_idx] = item1
      self.items['2'][self.buffer_idx] = item2
      self.items['3'][self.buffer_idx] = item3
      self.items['4'][self.buffer_idx] = item4
    self.buffer_idx += 1
    return 

  def sample(self, n:int):
    idxes = [random.randint(0, len(self._buffer)-1) for _ in range(n)]
    return  self.items['0'][idxes],\
            self.items['1'][idxes],\
            self.items['2'][idxes],\
            self.items['3'][idxes],\
            self.items['4'][idxes]

