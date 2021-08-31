# https://keras.io/examples/rl/deep_q_network_breakout/
# https://github.com/rlcode/reinforcement-learning-kr/blob/master/3-atari/1-breakout/breakout_dqn.py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import gym
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input, ReLU, Conv2D, Flatten
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
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

scores_avg, scores_raw, epsilons, losses, loss_list, score_avg, end = [], [], [], [], [], 0, False
FILENAME = "BreakoutDeterministic-v4_DQN"
def save_statistics():
  # View data
  plt.clf()
  plt.subplot(311)
  plt.plot(scores_avg, 'b')
  plt.plot(scores_raw, 'b', alpha=0.8, linewidth=0.5)
  plt.xlabel('Episodes'); plt.ylabel('average score'); plt.grid()
  plt.title(FILENAME)
  plt.subplot(312)
  plt.plot(epsilons, 'b')
  plt.xlabel('Episodes'); plt.ylabel('epsilon'); plt.grid()
  plt.subplot(313)
  plt.plot(losses, 'b')
  plt.xlabel('Episodes'); plt.ylabel('losses') ;plt.grid()
  plt.savefig(FILENAME + "_TF.jpg", dpi=100)

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
      "TRAIN_START":20000,
      "MEMORY_SIZE":100000,
    },
    "TRAIN_FREQ":4,
    "UPDATE_FREQ":2000,
  },
}
ENV_NAME = cfg['ENV']['NAME']
if __name__ == "__main__":
  env = gym.make(ENV_NAME)
  print('States ',env.observation_space, env.observation_space.shape,', Actions ', env.action_space, env.action_space.n)
  agent = DQNAgent(env, cfg)
  featurization = Featurization(cfg['ENV']['IMG_SIZE'])
  global_steps = 0
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
      action = agent.get_actions(feature)
      observe, reward, done, info = env.step(action=action)
      next_feature = featurization.preprocessing(observe)
      agent.remember(feature, action, reward, next_feature, done)
      loss = agent.train()
      agent.update_target_net()

      episode_score += reward
      episode_step += 1
      global_steps += 1
      feature = next_feature
      loss_list.append(loss)
      # break
      if (done == True) or (episode_step > MAX_STEP_PER_EPISODE):
        score_avg = 0.9 * score_avg + 0.1 * episode_score if score_avg != 0 else episode_score
        print('{:6d} epi with {:8d} steps, epi score {:5.1f}, score_avg {:10.5f}'.format(e+1,global_steps,episode_score, score_avg))
        scores_avg.append(score_avg)
        scores_raw.append(episode_score)
        losses.append(np.mean(loss_list))
        epsilons.append(agent.epsilon)
        if e % SAVE_FREQ == 0:
          save_statistics()
        if score_avg > END_SCORE:
          agent.save_model("")
          save_statistics()
          end = True
        break
    if end == True:
      env.close()
      print("End")
      break