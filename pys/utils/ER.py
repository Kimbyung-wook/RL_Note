import numpy as np
import random
from tictoc import TicToc
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

class ReplayMemory_openai():
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

  # def append(self, obs, action, reward, next_obs, done)->None:
  #   transition = (obs, action, reward, next_obs, done)
  #   self.buffer_idx = self.buffer_idx % self._capacity
  #   if(len(self._buffer) < self._capacity):
  #     self._buffer.append(transition)
  #   else:
  #     self._buffer[self.buffer_idx] = transition
  #   self.buffer_idx += 1

  def append(self, transition:list)->None:
    self._buffer_idx = self._buffer_idx % self._capacity
    if(len(self._buffer) < self._capacity):
      self._buffer.append(transition)
    else:
      self._buffer[self._buffer_idx] = transition
    self._buffer_idx += 1

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
      0:None,
      1:None,
      2:None,
      3:None,
      4:None,
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
      self.items[0] = item0
      self.items[1] = item1
      self.items[2] = item2
      self.items[3] = item3
      self.items[4] = item4
    elif self.buffer_idx < self.capacity: # Buffer is not full
      self.items[0] = np.append(self.items[0], item0, axis=0) 
      self.items[1] = np.append(self.items[1], item1, axis=0) 
      self.items[2] = np.append(self.items[2], item2, axis=0) 
      self.items[3] = np.append(self.items[3], item3, axis=0) 
      self.items[4] = np.append(self.items[4], item4, axis=0) 
    else: # Buffer is full
      idx = self.buffer_idx % self.capacity
      self.items[0][idx] = item0
      self.items[1][idx] = item1
      self.items[2][idx] = item2
      self.items[3][idx] = item3
      self.items[4][idx] = item4
    self.buffer_idx += 1
    return 

  def sample(self, n:int):
    idxes = [random.randint(0, len(self.items[0])-1) for _ in range(n)]
    return  self.items[0][idxes],\
            self.items[1][idxes],\
            self.items[2][idxes],\
            self.items[3][idxes],\
            self.items[4][idxes]

class ReplayMemory_nparray5_predefined():
  def __init__(self, capacity:int)->None:
    self.items = {
      0:None,
      1:None,
      2:None,
      3:None,
      4:None,
    }
    self.buffer_idx = 0
    self.capacity = capacity
    self.memory_size = self.buffer_idx
    self.is_first = True
    return 
  
  def append(self, transition:list)->None:
    if self.is_first == True:
      for i in range(len(self.items)):
        shapes = np.shape(transition[i])
        shapes = list(shapes).insert(0,self.capacity)
        self.items[i] = np.zeros(shapes)
    print(np.shape(transition))
    print(np.shape(self.items[0]))
    print(np.shape(np.array([transition[0]])))
    idx = self.buffer_idx % self.capacity
    # Error!!
    self.items[0][idx] = transition[0]
    self.items[1][idx] = transition[1]
    self.items[2][idx] = transition[2]
    self.items[3][idx] = transition[3]
    self.items[4][idx] = transition[4]
    self.buffer_idx += 1
    self.memory_size = max(self.capacity, self.buffer_idx)
    return 

  def sample(self, n:int):
    idxes = [random.randint(0, self.memory_size) for _ in range(n)]
    return  self.items[0][idxes],\
            self.items[1][idxes],\
            self.items[2][idxes],\
            self.items[3][idxes],\
            self.items[4][idxes]

if __name__ == "__main__":
  print('Memory Append/Sample Speed Test')
  memory_size = 10000
  append_size = 20000
  sample_size = 1000
  timer1 = TicToc(); 
  timer2 = TicToc(); 
  buffer1 = ReplayMemory(memory_size)
  timer1.tic()
  for i in range(append_size):
    item0 = np.array([i,0])
    item1 = np.array([i,1])
    item2 = np.array([i,2])
    item3 = np.array([i,3])
    item4 = np.array([i,4])
    transition = (item0, item1, item2, item3, item4)
    buffer1.append(transition)
  timer1.toc()
  timer2.tic()
  mini_batch = buffer1.sample(sample_size)
  states    = np.array([sample[0] for sample in mini_batch])
  actions   = np.array([sample[1] for sample in mini_batch])
  rewards   = np.array([sample[2] for sample in mini_batch])
  next_states = np.array([sample[3] for sample in mini_batch])
  dones     = np.array([sample[4] for sample in mini_batch])
  timer2.toc()
  print('ReplayMemory \t\tpush {:10.6f} sample {:10.6f}'.format(timer1.get_dt(), timer2.get_dt()))

  timer1 = TicToc(); 
  timer2 = TicToc(); 
  buffer1 = ReplayMemory_openai(memory_size)
  timer1.tic()
  for i in range(append_size):
    item0 = np.array([i,0])
    item1 = np.array([i,1])
    item2 = np.array([i,2])
    item3 = np.array([i,3])
    item4 = np.array([i,4])
    transition = (item0, item1, item2, item3, item4)
    buffer1.append(transition)
  timer1.toc()
  timer2.tic()
  states, actions, rewards, next_states, dones = buffer1.sample(sample_size)
  timer2.toc()
  print('ReplayMemory_openai \tpush {:10.6f} sample {:10.6f}'.format(timer1.get_dt(), timer2.get_dt()))

  timer1 = TicToc(); 
  timer2 = TicToc(); 
  buffer1 = ReplayMemory_nparray5(memory_size)
  timer1.tic()
  for i in range(append_size):
    item0 = np.array([i,0])
    item1 = np.array([i,1])
    item2 = np.array([i,2])
    item3 = np.array([i,3])
    item4 = np.array([i,4])
    transition = (item0, item1, item2, item3, item4)
    buffer1.append(transition)
  timer1.toc()
  timer2.tic()
  states, actions, rewards, next_states, dones = buffer1.sample(sample_size)
  timer2.toc()
  print('ReplayMemory_nparray5 \tpush {:10.6f} sample {:10.6f}'.format(timer1.get_dt(), timer2.get_dt()))


  timer1 = TicToc(); 
  timer2 = TicToc(); 
  buffer1 = ReplayMemory_nparray5_predefined(memory_size)
  timer1.tic()
  for i in range(append_size):
    item0 = np.array([i,0])
    item1 = np.array([i,1])
    item2 = np.array([i,2])
    item3 = np.array([i,3])
    item4 = np.array([i,4])
    transition = (item0, item1, item2, item3, item4)
    buffer1.append(transition)
  timer1.toc()
  timer2.tic()
  states, actions, rewards, next_states, dones = buffer1.sample(sample_size)
  timer2.toc()
  print('ReplayMemory_nparray5_predefined \tpush {:10.6f} sample {:10.6f}'.format(timer1.get_dt(), timer2.get_dt()))


