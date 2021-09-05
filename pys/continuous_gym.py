import os, sys
from utils.set_path import set_path
dir_name = 'RL_Note'
root_path = set_path(dir_name)
print(root_path)
workspace_path = root_path + '\pys'
print(workspace_path)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import gym
import argparse
import numpy as np
import matplotlib.pyplot as plt
from env_config  import env_configs
from wrapper.gym_wrapper import GymWrapper              # Env Wrapper
from agent.agent_broker import continuous_agent_broker  # Agent broker
from utils.gpu_memory_limiter import gpu_memory_limiter # GPU 
from configs.nn_cfg import *  # Network Model Configuration

gpu_memory_limiter(1024*2)
parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str,   default="MountainCarContinuous-v0")
parser.add_argument('--train',    type=str,   default='TRAIN')
args = parser.parse_args()

lists = (
  # ('SAC','ER'), ('SAC','PER'), ('SAC','HER'),\
  # ('TD3','ER'), ('TD3','PER'), ('TD3','HER'),\
  # ('DDPG','ER'),('DDPG','PER'), ('DDPG','HER'),\
  # ('DDPG', 'ER',('',),), # 
  # ('DDPG', 'PER',('',),), # 
  # ('SAC', 'ER',('FIXED',),),
  ('SAC','PER',('FIXED',),),
)
print('Batch list : ',lists)

# ENV_NAME = "Pendulum-v0"
# ENV_NAME = "LunarLanderContinuous-v2"
# ENV_NAME = "MountainCarContinuous-v0"
ENV_NAME = args.env_name
if __name__ == "__main__":
  for item in lists:
    cfg = {\
    "ENV":{
      "NAME":ENV_NAME,\
      'STATE':{
        # 'TYPE':('IMAGE',),
        # 'STATE_SPACE':((84,84,4),()),
        'TYPE':('ARRAY',),
      },
      # 'DEPTH_RANGE':(1.0,20.0),
      # 'INIT_POSITION':(-20.0,0.0,0.0),
    },
    "RL":{
      "ALGORITHM":item[0],\
      "TYPE":item[2],
      "NETWORK":classic_continuous_cfg
    },\
    "ER":{
      "ALGORITHM":item[1],\
      "REPLAY_N":8,\
      "STRATEGY":"EPISODE",\
      # "REWARD_FUNC":reward_function,\
      # "DONE_FUNC":done_function,\
    },\
    "BATCH_SIZE":8,\
    "TRAIN_START":2000,\
    "MEMORY_SIZE":50000,\
    "ADD_NAME":()
    }
    RL_NAME = cfg["RL"]["ALGORITHM"]
    for item1 in cfg['RL']['TYPE']:
      RL_NAME = RL_NAME + '_' + item1
    FILENAME = cfg["ENV"]['NAME'] + '_' + RL_NAME + '_' + cfg["ER"]["ALGORITHM"]
    if cfg["ER"]["ALGORITHM"] == "HER":
      FILENAME = FILENAME + '_' + cfg["ER"]["STRATEGY"]
    for item1 in cfg["ADD_NAME"]:
      FILENAME  = FILENAME + '_' + item1
    env_config = env_configs[cfg["ENV"]['NAME']]
    EPISODES  = env_config["EPISODES"]
    END_SCORE = env_config["END_SCORE"]

    # Define Environment
    env = gym.make(ENV_NAME)
    env = GymWrapper(env, cfg['ENV'])
    # Define RL Agent
    agent = continuous_agent_broker(rl=cfg["RL"]["ALGORITHM"], env=env, cfg=cfg)

    plt.clf()
    figure = plt.gcf()
    figure.set_size_inches(8,6)

    scores_avg, scores_raw, episodes, losses = [], [], [], []
    critic_mean, actor_mean = [], []
    score_avg = 0
    save_freq = 10
    end = False
    show_media_info = True
    goal = np.array([1.0,0.0,0.0])

    if args.train.upper() == 'TRAIN':
      for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset()
        critic_losses = []
        actor_losses = []
        while not done:
          # if e%100 == 0: env.render()
          # Interact with env.
          action = agent.get_action(state)
          next_state, reward, done, info = env.step(action)
          agent.remember(state, action, reward, next_state, done, goal)
          do_train, critic_loss, actor_loss, train_consuming_time = agent.train_model()
          agent.update_model(done)
          state = next_state
          # 
          score += reward
          critic_losses.append(critic_loss)
          actor_losses.append(actor_loss)
          if do_train == True:
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)
          if show_media_info:
            print("-------------- Variable shapes --------------")
            print("State Shape : ", np.shape(state))
            print("Action Shape : ", np.shape(action))
            print("Reward Shape : ", np.shape(reward))
            print("done Shape : ", np.shape(done))
            print("---------------------------------------------")
            show_media_info = False
          if done:
            score_avg = 0.9 * score_avg + 0.1 * score if score_avg != 0 else score
            print("episode: {0:3d} | score avg: {1:3.2f} | mem size {2:6d} |"
              .format(e, score_avg, len(agent.memory)))

            episodes.append(e)
            scores_avg.append(score_avg)
            scores_raw.append(score)
            critic_mean.append(np.mean(critic_losses))
            actor_mean.append(np.mean(actor_losses))
            if e % save_freq == 0:
              # View data
              plt.clf()
              plt.subplot(311)
              plt.plot(episodes, scores_avg, 'b')
              plt.plot(episodes, scores_raw, 'b', alpha=0.8, linewidth=0.5)
              plt.xlabel('episode'); plt.ylabel('average score'); plt.grid()
              plt.title(FILENAME)
              plt.subplot(312)
              plt.plot(episodes, critic_mean, 'b.',markersize=3)
              plt.xlabel('episode'); plt.ylabel('critic loss'); plt.grid()
              plt.subplot(313)
              plt.plot(episodes, actor_mean, 'b.',markersize=3)
              plt.xlabel('episode'); plt.ylabel('actor loss'); plt.grid()
              plt.savefig(workspace_path + "\\result\\img\\" + FILENAME + "_TF.jpg", dpi=100)

            # 이동 평균이 0 이상일 때 종료
            if score_avg > END_SCORE:
              agent.save_model(workspace_path + "\\result\\save_model\\")
              # View data
              plt.clf()
              plt.subplot(311)
              plt.plot(episodes, scores_avg, 'b')
              plt.plot(episodes, scores_raw, 'b', alpha=0.8, linewidth=0.5)
              plt.xlabel('episode'); plt.ylabel('average score'); plt.grid()
              plt.title(FILENAME)
              plt.subplot(312)
              plt.plot(episodes, critic_mean, 'b.',markersize=3)
              plt.xlabel('episode'); plt.ylabel('critic loss'); plt.grid()
              plt.subplot(313)
              plt.plot(episodes, actor_mean, 'b.',markersize=3)
              plt.xlabel('episode'); plt.ylabel('actor loss'); plt.grid()
              plt.savefig(workspace_path + "\\result\\img\\" + FILENAME + "_TF.jpg", dpi=100)
              end = True
              break
        if end == True:
          np.save(workspace_path + "\\result\\data\\" + FILENAME + "_TF_scores_avg",scores_avg)
          np.save(workspace_path + "\\result\\data\\" + FILENAME + "_TF_scores_raw",scores_raw)
          np.save(workspace_path + "\\result\\data\\" + FILENAME + "_TF_losses",losses)
          break

    elif args.train.upper() == 'TEST': # Test
      agent.load_model(workspace_path + "\\result\\save_model\\")
      for e in range(10):
        # Episode initialization
        done = False; state = env.reset(); score = 0
        while not done:
          env.render()
          action = agent.get_action(state)
          next_state, reward, done, info = env.step(action)
          score += reward
          state = next_state
          if done:
            score_avg = 0.9 * score_avg + 0.1 * score if score_avg != 0 else score
            print("episode: {0:3d} | score avg: {1:3.2f} |"
              .format(e, score_avg))
    env.close()
    print("End")
