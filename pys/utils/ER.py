import numpy as np
import random
import time
from tictoc import TicToc
show_memory_size = False
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

class ReplayMemory_openai_predefined():
  '''
  This is OpenAI baseline style-Replay Memory
  Refer from
  https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
  '''
  def __init__(self, capacity:int)->None:
    self._buffer = []
    self._buffer_idx = 0
    self._capacity = capacity
    self._memory_size = 0
    self.is_first = True
    return

  def append(self, transition:list)->None:
    if self.is_first==True:
      for _ in range(self._capacity):
        self._buffer.append(transition)

    self._buffer_idx = self._buffer_idx % self._capacity
    self._buffer[self._buffer_idx] = transition
    self._buffer_idx += 1
    self._memory_size = min(self._capacity, self._buffer_idx)

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
    idxes = [random.randint(0, self._memory_size-1) for _ in range(n)]
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
  
  # def append(self, transition:list)->None:
  #   item0 = np.array([transition[0]])
  #   item1 = np.array([transition[1]])
  #   item2 = np.array([transition[2]])
  #   item3 = np.array([transition[3]])
  #   item4 = np.array([transition[4]])
  #   if self.buffer_idx is 0: # First append
  #     self.items[0] = item0
  #     self.items[1] = item1
  #     self.items[2] = item2
  #     self.items[3] = item3
  #     self.items[4] = item4
  #   elif self.buffer_idx < self.capacity: # Buffer is not full
  #     self.items[0] = np.append(self.items[0], item0, axis=0) 
  #     self.items[1] = np.append(self.items[1], item1, axis=0) 
  #     self.items[2] = np.append(self.items[2], item2, axis=0) 
  #     self.items[3] = np.append(self.items[3], item3, axis=0) 
  #     self.items[4] = np.append(self.items[4], item4, axis=0) 
  #   else: # Buffer is full
  #     idx = self.buffer_idx % self.capacity
  #     self.items[0][idx] = item0
  #     self.items[1][idx] = item1
  #     self.items[2][idx] = item2
  #     self.items[3][idx] = item3
  #     self.items[4][idx] = item4
  #   self.buffer_idx += 1
  #   return 
  
  def append(self, item0, item1, item2, item3, item4)->None:
    item0 = np.array([item0])
    item1 = np.array([item1])
    item2 = np.array([item2])
    item3 = np.array([item3])
    item4 = np.array([item4])
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

class ReplayMemory_nparray5_predefined_idx():
  def __init__(self, capacity:int)->None:
    self.items = {
      0:None,
      1:None,
      2:None,
      3:None,
      4:None,
    }
    self._buffer_idx = 0
    self._capacity = capacity
    self._memory_size = self._buffer_idx
    self._is_first = True
    return 
  
  def append(self, transition:list)->None:
    if self._is_first == True:
      for i in range(len(self.items)):
        shape = np.shape(transition[i])
        shapes= [self._capacity]
        shapes.extend(shape)
        self.items[i] = np.zeros(shapes)
      self._is_first = False
      if show_memory_size == True :
        print('memory alloc : ',np.shape(self.items[0]),\
                                np.shape(self.items[1]),\
                                np.shape(self.items[2]),\
                                np.shape(self.items[3]),\
                                np.shape(self.items[4]))
    idx = self._buffer_idx % self._capacity
    self.items[0][idx] = np.expand_dims(transition[0],axis=0)
    self.items[1][idx] = np.expand_dims(transition[1],axis=0)
    self.items[2][idx] = np.expand_dims(transition[2],axis=0)
    self.items[3][idx] = np.expand_dims(transition[3],axis=0)
    self.items[4][idx] = np.expand_dims(transition[4],axis=0)
    self._buffer_idx += 1
    self._memory_size = min(self._capacity, self._buffer_idx)
    return 

  def sample(self, n:int):
    idxes = [random.randint(0, self._memory_size-1) for _ in range(n)]
    return  self.items[0][idxes],\
            self.items[1][idxes],\
            self.items[2][idxes],\
            self.items[3][idxes],\
            self.items[4][idxes]

class ReplayMemory_nparray5_predefined_stack():
  def __init__(self, capacity:int)->None:
    self.items = {
      0:None,
      1:None,
      2:None,
      3:None,
      4:None,
    }
    self._buffer_idx = 0
    self._capacity = capacity
    self._memory_size = self._buffer_idx
    self._is_first = True
    return 
  
  def append(self, transition:list)->None:
    if self._is_first == True:
      for i in range(len(self.items)):
        shape = np.shape(transition[i])
        shapes= [self._capacity]
        shapes.extend(shape)
        self.items[i] = np.zeros(shapes)
      self._is_first = False
      if show_memory_size == True :
        print('memory alloc : ',np.shape(self.items[0]),\
                                np.shape(self.items[1]),\
                                np.shape(self.items[2]),\
                                np.shape(self.items[3]),\
                                np.shape(self.items[4]))
    idx = self._buffer_idx % self._capacity
    self.items[0][idx] = np.expand_dims(transition[0],axis=0)
    self.items[1][idx] = np.expand_dims(transition[1],axis=0)
    self.items[2][idx] = np.expand_dims(transition[2],axis=0)
    self.items[3][idx] = np.expand_dims(transition[3],axis=0)
    self.items[4][idx] = np.expand_dims(transition[4],axis=0)
    self._buffer_idx += 1
    self._memory_size = min(self._capacity, self._buffer_idx)
    return 

  def sample(self, n:int):
    is_first = True
    idxes = [random.randint(0, self._memory_size-1) for _ in range(n)]

    item1 = np.array([],dtype=np.float32)
    item2 = np.array([],dtype=np.float32)
    item3 = np.array([],dtype=np.float32)
    item4 = np.array([],dtype=np.float32)
    item5 = np.array([],dtype=np.float32)
    for idx in idxes:
      if is_first == True:
        # print('idx : ',idx)
        # print(self.items[0][idx])
        # item1 = np.hstack((item1, self.items[0][idx]))
        # item2 = np.hstack((item2, self.items[1][idx]))
        # item3 = np.hstack((item3, self.items[2][idx]))
        # item4 = np.hstack((item4, self.items[3][idx]))
        # item5 = np.hstack((item5, self.items[4][idx]))
        item1 = np.expand_dims(self.items[0][idx],axis=0)
        item2 = np.expand_dims(self.items[1][idx],axis=0)
        item3 = np.expand_dims(self.items[2][idx],axis=0)
        item4 = np.expand_dims(self.items[3][idx],axis=0)
        item5 = np.expand_dims(self.items[4][idx],axis=0)
        # print(np.shape(item1))
        is_first = False
      else:
        item1 = np.dstack((item1, np.expand_dims(self.items[0][idx],axis=0)))
        item2 = np.dstack((item2, np.expand_dims(self.items[1][idx],axis=0)))
        item3 = np.dstack((item3, np.expand_dims(self.items[2][idx],axis=0)))
        item4 = np.dstack((item4, np.expand_dims(self.items[3][idx],axis=0)))
        item5 = np.dstack((item5, np.expand_dims(self.items[4][idx],axis=0)))

    return item1, item2, item3, item4, item5
    # return  self.items[0][idxes],\
    #         self.items[1][idxes],\
    #         self.items[2][idxes],\
    #         self.items[3][idxes],\
    #         self.items[4][idxes]

class ReplayMemory_nparray5_predefined_concat():
  def __init__(self, capacity:int)->None:
    self.items = {
      0:None,
      1:None,
      2:None,
      3:None,
      4:None,
    }
    self._buffer_idx = 0
    self._capacity = capacity
    self._memory_size = self._buffer_idx
    self._is_first = True
    return 
  
  def append(self, transition:list)->None:
    if self._is_first == True:
      for i in range(len(self.items)):
        shape = np.shape(transition[i])
        shapes= [self._capacity]
        shapes.extend(shape)
        self.items[i] = np.zeros(shapes)
      self._is_first = False
      if show_memory_size == True :
        print('memory alloc : ',np.shape(self.items[0]),\
                                np.shape(self.items[1]),\
                                np.shape(self.items[2]),\
                                np.shape(self.items[3]),\
                                np.shape(self.items[4]))
    idx = self._buffer_idx % self._capacity
    self.items[0][idx] = np.expand_dims(transition[0],axis=0)
    self.items[1][idx] = np.expand_dims(transition[1],axis=0)
    self.items[2][idx] = np.expand_dims(transition[2],axis=0)
    self.items[3][idx] = np.expand_dims(transition[3],axis=0)
    self.items[4][idx] = np.expand_dims(transition[4],axis=0)
    self._buffer_idx += 1
    self._memory_size = min(self._capacity, self._buffer_idx)
    return 

  def sample(self, n:int):
    is_first = True
    idxes = [random.randint(0, self._memory_size-1) for _ in range(n)]

    item1 = np.array([],dtype=np.float32)
    item2 = np.array([],dtype=np.float32)
    item3 = np.array([],dtype=np.float32)
    item4 = np.array([],dtype=np.float32)
    item5 = np.array([],dtype=np.float32)
    for idx in idxes:
      if is_first == True:
        # print('idx : ',idx)
        # print(self.items[0][idx])
        # item1 = np.hstack((item1, self.items[0][idx]))
        # item2 = np.hstack((item2, self.items[1][idx]))
        # item3 = np.hstack((item3, self.items[2][idx]))
        # item4 = np.hstack((item4, self.items[3][idx]))
        # item5 = np.hstack((item5, self.items[4][idx]))
        item1 = np.expand_dims(self.items[0][idx],axis=0)
        item2 = np.expand_dims(self.items[1][idx],axis=0)
        item3 = np.expand_dims(self.items[2][idx],axis=0)
        item4 = np.expand_dims(self.items[3][idx],axis=0)
        item5 = np.expand_dims(self.items[4][idx],axis=0)
        is_first = False
      else:
        item1 = np.concatenate((item1, np.expand_dims(self.items[0][idx],axis=0)), axis=0)
        item2 = np.concatenate((item2, np.expand_dims(self.items[1][idx],axis=0)), axis=0)
        item3 = np.concatenate((item3, np.expand_dims(self.items[2][idx],axis=0)), axis=0)
        item4 = np.concatenate((item4, np.expand_dims(self.items[3][idx],axis=0)), axis=0)
        item5 = np.concatenate((item5, np.expand_dims(self.items[4][idx],axis=0)), axis=0)

    return item1, item2, item3, item4, item5
    # return  self.items[0][idxes],\
    #         self.items[1][idxes],\
    #         self.items[2][idxes],\
    #         self.items[3][idxes],\
    #         self.items[4][idxes]

if __name__ == "__main__":
  print('Memory Append/Sample Speed Test')
  multiple = 100
  memory_size = multiple * 1000
  append_size = multiple * 2000
  sample_size = multiple * 100
  print('Test condition : Get running time about')
  print('    - push time')
  print('    - pop time')
  print('    - memory pre-defining')
  print('---        Memory size : {:8.1f} k'.format(memory_size/1000.0))
  print('--- Appended data size : {:8.1f} k'.format(append_size/1000.0))
  print('---      Sampling size : {:8.1f} k'.format(sample_size/1000.0))
  i = 1
  data0 = np.array([[i,0],[i*i,i*i*i]])
  data1 = np.random.normal(size=(6,4,4))
  data2 = np.random.normal(size=(5))
  data3 = np.random.normal(size=(1))
  data4 = np.random.normal(size=(1))
  data5 = np.random.normal(size=(3))
  # data0 = np.array([[i,0],[i*i,i*i*i]])
  # data1 = np.random.normal(size=(6,64,64))
  # data2 = np.random.normal(size=(6,64,64))
  # data3 = np.random.normal(size=(6,64,64))
  # data4 = np.random.normal(size=(6,64,64))
  # data5 = np.random.normal(size=(6,64,64))
  print('Shape of Input data1', np.shape(data1))
  print('Shape of Input data2', np.shape(data2))
  print('Shape of Input data3', np.shape(data3))
  print('Shape of Input data4', np.shape(data4))
  print('Shape of Input data5', np.shape(data5))
  timer1 = TicToc(); 
  timer2 = TicToc(); 
  buffer1 = ReplayMemory(memory_size)
  timer1.tic()
  for i in range(append_size):
    item0 = data0;  item1 = data1;  item2 = data2;  item3 = data3;  item4 = data4
    transition = (item0, item1, item2, item3, item4)
    buffer1.append(transition)
  timer1.toc()
  timer2.tic()
  mini_batch = buffer1.sample(sample_size)
  states      = np.array([sample[0] for sample in mini_batch])
  actions     = np.array([sample[1] for sample in mini_batch])
  rewards     = np.array([sample[2] for sample in mini_batch])
  next_states = np.array([sample[3] for sample in mini_batch])
  dones       = np.array([sample[4] for sample in mini_batch])
  timer2.toc()
  # print('shape of states ',np.shape(states))
  print('Test1   : push {:10.6f} sample {:10.6f} - ReplayMemory'.format(timer1.get_dt(), timer2.get_dt()))
  del(buffer1)
  time.sleep(2.0)

  buffer1 = ReplayMemory_openai(memory_size)
  timer1.tic()
  for i in range(append_size):
    item0 = data0;  item1 = data1;  item2 = data2;  item3 = data3;  item4 = data4
    transition = (item0, item1, item2, item3, item4)
    buffer1.append(transition)
  timer1.toc()
  timer2.tic()
  states, actions, rewards, next_states, dones = buffer1.sample(sample_size)
  timer2.toc()
  # print('shape of states ',np.shape(states))
  print('Test2-1 : push {:10.6f} sample {:10.6f} - ReplayMemory_openai'.format(timer1.get_dt(), timer2.get_dt()))
  del(buffer1)
  time.sleep(2.0)

  ## Making predefined list is too slow
  # buffer1 = ReplayMemory_openai_predefined(memory_size)
  # timer1.tic()
  # for i in range(append_size):
  #   item0 = np.array([[i,0],[i*i,i*i*i]])
  #   item1 = np.array([[i,0],[i*i,i*i*i]])
  #   item2 = np.array([[i,0],[i*i,i*i*i]])
  #   item3 = np.array([[i,0],[i*i,i*i*i]])
  #   item4 = np.array([[i,0],[i*i,i*i*i]])
  #   transition = (item0, item1, item2, item3, item4)
  #   buffer1.append(transition)
  # timer1.toc()
  # timer2.tic()
  # states, actions, rewards, next_states, dones = buffer1.sample(sample_size)
  # timer2.toc()
  # print('Test2-2 : push {:10.6f} sample {:10.6f} - ReplayMemory_openai_predefined'.format(timer1.get_dt(), timer2.get_dt()))

  ## Make new memory on numpy is too slow
  # buffer1 = ReplayMemory_nparray5(memory_size)
  # timer1.tic()
  # for i in range(append_size):
  #   item0 = np.array([[i,0],[i*i,i*i*i]])
  #   item1 = np.array([[i,0],[i*i,i*i*i]])
  #   item2 = np.array([[i,0],[i*i,i*i*i]])
  #   item3 = np.array([[i,0],[i*i,i*i*i]])
  #   item4 = np.array([[i,0],[i*i,i*i*i]])
  #   # transition = (item0, item1, item2, item3, item4)
  #   # buffer1.append(transition)
  #   buffer1.append(item0, item1, item2, item3, item4)
  # timer1.toc()
  # timer2.tic()
  # states, actions, rewards, next_states, dones = buffer1.sample(sample_size)
  # timer2.toc()
  # print('Test3   : push {:10.6f} sample {:10.6f} ReplayMemory_nparray5'.format(timer1.get_dt(), timer2.get_dt()))
  # del(buffer1)
  # time.sleep(2.0)

  timer3 = TicToc()
  timer3.tic()
  buffer1 = ReplayMemory_nparray5_predefined_idx(memory_size)
  item0 = data0;  item1 = data1;  item2 = data2;  item3 = data3;  item4 = data4
  transition = (item0, item1, item2, item3, item4)
  buffer1.append(transition)
  timer3.toc()
  timer1.tic()
  for i in range(append_size):
    item0 = data0;  item1 = data1;  item2 = data2;  item3 = data3;  item4 = data4
    transition = (item0, item1, item2, item3, item4)
    buffer1.append(transition)
  timer1.toc()
  timer2.tic()
  states, actions, rewards, next_states, dones = buffer1.sample(sample_size)
  timer2.toc()
  # print('shape of states ',np.shape(states))
  print('Test4-1 : push {:10.6f} sample {:10.6f} predef {:10.6f} - ReplayMemory_nparray5_predefined_idx'.format(timer1.get_dt(), timer2.get_dt(), timer3.get_dt()))
  del(buffer1)
  time.sleep(2.0)

  # buffer1 = ReplayMemory_nparray5_predefined_stack(memory_size)
  # timer3.tic()
  # item0 = data0;  item1 = data1;  item2 = data2;  item3 = data3;  item4 = data4
  # transition = (item0, item1, item2, item3, item4)
  # buffer1.append(transition)
  # timer3.toc()
  # timer1.tic()
  # for i in range(append_size):
  #   item0 = data0;  item1 = data1;  item2 = data2;  item3 = data3;  item4 = data4
  #   transition = (item0, item1, item2, item3, item4)
  #   buffer1.append(transition)
  # timer1.toc()
  # timer2.tic()
  # states, actions, rewards, next_states, dones = buffer1.sample(sample_size)
  # timer2.toc()
  # # print('shape of states ',np.shape(states))
  # print('Test4-2 : push {:10.6f} sample {:10.6f} predef {:10.6f} - ReplayMemory_nparray5_predefined_stack'.format(timer1.get_dt(), timer2.get_dt(), timer3.get_dt()))
  # del(buffer1)
  # time.sleep(2.0)

  # buffer1 = ReplayMemory_nparray5_predefined_concat(memory_size)
  # timer3.tic()
  # item0 = data0;  item1 = data1;  item2 = data2;  item3 = data3;  item4 = data4
  # transition = (item0, item1, item2, item3, item4)
  # buffer1.append(transition)
  # timer3.toc()
  # timer1.tic()
  # for i in range(append_size):
  #   item0 = data0;  item1 = data1;  item2 = data2;  item3 = data3;  item4 = data4
  #   transition = (item0, item1, item2, item3, item4)
  #   buffer1.append(transition)
  # timer1.toc()
  # timer2.tic()
  # states, actions, rewards, next_states, dones = buffer1.sample(sample_size)
  # timer2.toc()
  # # print('shape of states ',np.shape(states))
  # print('Test4-3 : push {:10.6f} sample {:10.6f} predef {:10.6f} - ReplayMemory_nparray5_predefined_concat'.format(timer1.get_dt(), timer2.get_dt(), timer3.get_dt()))
  # del(buffer1)
  # time.sleep(2.0)


