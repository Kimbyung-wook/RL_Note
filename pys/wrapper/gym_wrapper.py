import os, sys
cwd = os.getcwd()
dir_name = 'RL_Note'
tmp1 = cwd.lower()
tmp2 = dir_name.lower()
pos = tmp1.find(tmp2)
root_path = cwd[0:pos] + dir_name
sys.path.append(root_path)

import gym
import random
import numpy as np
import matplotlib.pyplot as plt
from utils.shortsmaker import ShortsMaker

class GymWrapper():
  def __init__(self, env:object, wrapper_cfg:dict)->object:
    self.env = env
    self.observation_space = self.env.observation_space
    self.action_space = self.env.action_space
    self.cfg = wrapper_cfg
    self.state_type = self.cfg['STATE']['TYPE'] # IMAGE, ARRAY
    self.state = None
    self.show_video_info = True
    if 'IMAGE' in self.state_type:
      self.state_space= self.cfg['STATE']['STATE_SPACE']
      self.shortmaker = ShortsMaker(self.state_space[0])

  def reset(self):
    obs = self.env.reset()
    state = obs
    if 'IMAGE' in self.state_type:
      img = self.env.render(mode='rgb_array')
      if self.show_video_info:
        print('Displayed image shape ',np.shape(img)); self.show_video_info = False
      video = self.shortmaker.get_video(img)
      state = video
    return state

  def step(self, action):
    next_obs, reward, done, info = self.env.step(action)
    next_state = next_obs
    if 'IMAGE' in self.state_type:
      img = self.env.render(mode='rgb_array')
      next_video = self.shortmaker.get_video(img)
      next_state = next_video
    
    return next_state, reward, done, info
    
  def render(self):
    self.env.render()
    return 

  def close(self):
    self.env.close()
    return 
if __name__ == "__main__":
  # ENV_NAME = "Pendulum-v0"
  # ENV_NAME = "LunarLanderContinuous-v2"
  # ENV_NAME = "MountainCarContinuous-v0"
  # python -m atari_py.import_roms
  ENV_NAME = "BreakoutDeterministic-v4"
  wrapper_cfg={
    'NAME' : ENV_NAME,
    'STATE' : {
      'TYPE' : ('IMAGE',),
      'STATE_SPACE' : ((84,84,4),()),
      # 'TYPE':('ARRAY',),
    }
  }
  env = gym.make(ENV_NAME)
  env = GymWrapper(env, wrapper_cfg)
  show_media_info = True
  # fig = plt.figure(2)
  for e in range(5):
    state = env.reset()
    done = False
    while not done:
      # env.render()
      # plt.imshow(state, cmap='gray')
      # plt.show()
      action = env.action_space.sample()
      next_state, reward, done, info = env.step(action)
      state = next_state
      if show_media_info:
        print("-------------- Variable shapes --------------")
        print("State Shape : ", np.shape(state))
        print("Action Shape : ", np.shape(action))
        print("Reward Shape : ", np.shape(reward))
        print("done Shape : ", np.shape(done))
        print("---------------------------------------------")
        show_media_info = False


