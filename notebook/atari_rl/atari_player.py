# https://keras.io/examples/rl/deep_q_network_breakout/
# https://github.com/rlcode/reinforcement-learning-kr/blob/master/3-atari/1-breakout/breakout_dqn.py
# https://colab.research.google.com/github/GiannisMitr/DQN-Atari-Breakout/blob/master/dqn_atari_breakout.ipynb
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import gym, random, time
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input, ReLU, Conv2D, Flatten
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from ER import ReplayMemory
from PER import ProportionalPrioritizedMemory
from featurization import Featurization
from dqn_agent import DQNAgent

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(\
        gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*3)])
  except RuntimeError as e:
    print(e)

EPISODES = 10000
MAX_STEP_PER_EPISODE = 10000
END_SCORE = 40
SAVE_FREQ = 10
cfg = {\
  "ENV":{
    "NAME":"BreakoutDeterministic-v4",
    "IMG_SIZE":(84,84,4)
  },
  "RL":{
    "ALGORITHM":'DQN',
    "ER":{
      "ALGORITHM":'ER',
      "BATCH_SIZE":64,
      "TRAIN_START":10000,
      "MEMORY_SIZE":100000,
    },
    "TRAIN_FREQ":4,
    "UPDATE_FREQ":2000,
  },
}
ENV_NAME = cfg['ENV']['NAME']
FILENAME = cfg['ENV']['NAME'] + '_', cfg['RL']['ALGORITHM']
if __name__ == "__main__":
  # Log Setting
  dir_path = str(os.path.abspath('')) + '/result/' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
  model_save_dir = dir_path + '/save_model/'
  summary_writer = SummaryWriter(dir_path+'/tensorboard/')
  env = gym.make(ENV_NAME)
  agent = DQNAgent(env, cfg)
  featurization = Featurization(cfg['ENV']['IMG_SIZE'])
  global_steps = 0; score_avg = 0; end=False
  print('States ',env.observation_space, env.observation_space.shape,', Actions ', env.action_space, env.action_space.n)
  
  for e in range(EPISODES):
    observe = env.reset()
    feature = featurization.preprocessing(observe)
    episode_score = 0
    episode_step = 0
    loss_list = []
    while True:
      if e % 100 == 0:
        env.render(mode='human')
      # action = env.action_space.sample()
      action = agent.get_action(feature)
      observe, reward, done, info = env.step(action=action)
      next_feature = featurization.preprocessing(observe)
      agent.remember(feature, action, reward, next_feature, done)
      loss = agent.train_model()
      agent.update_model()

      episode_score += reward
      episode_step += 1; global_steps += 1
      feature = next_feature

      loss_list.append(loss)
      summary_writer.add_scalar('step/loss', loss, global_steps)
      summary_writer.add_scalar('step/action', action, global_steps)
      # break
      if (done == True) or (episode_step > MAX_STEP_PER_EPISODE):
        score_avg = 0.9 * score_avg + 0.1 * episode_score if score_avg != 0 else episode_score
        print('{:6d} epi with {:8d} steps, epi score {:5.1f}, score_avg {:10.5f}'.format(e+1,global_steps,episode_score, score_avg))
        
        summary_writer.add_scalar('epi/score_avg', score_avg, e)
        summary_writer.add_scalar('epi/score', episode_score, e)
        summary_writer.add_scalar('epi/loss_mean', np.mean(loss_list), e)
        summary_writer.add_scalar('epi/epsilon', agent.epsilon, e)
        if score_avg > END_SCORE:
          agent.save_model("")
          end = True
        break
    if end == True:
      env.close()
      print("End")
      break