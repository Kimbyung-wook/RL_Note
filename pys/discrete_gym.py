import os, sys
from utils.set_path import set_path
dir_name = 'RL_Note'
root_path = set_path(dir_name)
print(root_path)
workspace_path = root_path + '\pys'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import gym
import argparse
import numpy as np
import matplotlib.pyplot as plt
from env_config  import env_configs
from wrapper.gym_wrapper import GymWrapper              # Env Wrapper
from agent.agent_broker import discrete_agent_broker    # Agent broker
from utils.gpu_memory_limiter import gpu_memory_limiter # GPU 
from configs.nn_cfg import *  # Network Model Configuration

gpu_memory_limiter(1024)
parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str,   default="CartPole-v1")
parser.add_argument('--train',    type=str,   default='TRAIN')
args = parser.parse_args()
    
lists = (
    # ('DQN', 'ER',('',),),
    ('DQN','PER',('',),),
    # ('DQN', 'ER',('DUELING',),),
    # ('DQN','PER',('DUELING',),),
    # ('DQN', 'ER',('Q_penalty',),),
    # ('DQN','PER',('Q_penalty',),),
    # ('DQN','PER',('DUELING','Q_penalty',),),
    # ('MDQN','ER',''),
    # ('MDQN','PER',''),
    # ('MDQN','ER','DUELING',),
    # ('MDQN','PER','DUELING',),
  )
print('Batch list : ',lists)
print(args.train)
# ENV_NAME="Pong-v0"
# ENV_NAME="MountainCar-v0"
# ENV_NAME="LunarLander-v2"
# ENV_NAME="CartPole-v1"
ENV_NAME = args.env_name
if __name__ == "__main__":
  for item in lists:
    cfg = {\
      "ENV":{
        'NAME':ENV_NAME,
        'STATE':{
          # 'TYPE':('IMAGE',),
          # 'STATE_SPACE':((84,84,4),()),
          'TYPE':('ARRAY',),
        },
      },
      "RL":{
        "ALGORITHM":item[0],\
        "TYPE":item[2],
        # "TYPE":(item[2],'Q_penalty'),
        "NETWORK":classic_discrete_cfg
      },
      "ER":{
        "ALGORITHM":item[1],\
        "REPLAY_N":8,\
        "STRATEGY":"EPISODE",\
        # "REWARD_FUNC":reward_function,\
        # "DONE_FUNC":done_function,\
      },\
      "BATCH_SIZE":64,\
      "TRAIN_START":1000,\
      "MEMORY_SIZE":100000,\
      "ADD_NAME":()
    }
    RL_NAME = cfg["RL"]["ALGORITHM"]
    for item in cfg['RL']['TYPE']:
      RL_NAME = RL_NAME + '_' + item
    FILENAME = cfg["ENV"]['NAME'] + '_' + RL_NAME + '_' + cfg["ER"]["ALGORITHM"]
    if cfg["ER"]["ALGORITHM"] == "HER":
      FILENAME = FILENAME + '_' + cfg["ER"]["STRATEGY"]
    for item in cfg["ADD_NAME"]:
      FILENAME  = FILENAME + '_' + item
    env_config = env_configs[cfg["ENV"]['NAME']]
    EPISODES  = env_config["EPISODES"]
    END_SCORE = env_config["END_SCORE"]

    # Define Environment
    env = gym.make(cfg["ENV"]['NAME'])
    env = GymWrapper(env, cfg['ENV'])
    # Define RL Agent
    agent = discrete_agent_broker(rl=cfg["RL"]["ALGORITHM"], env=env, cfg=cfg)

    plt.clf()
    figure = plt.gcf()
    figure.set_size_inches(8,6)

    scores_avg, scores_raw, epsilons, losses = [], [], [], []
    score_avg = 0
    save_freq = 10
    global_steps = 0
    end = False
    show_media_info = True
    goal = (0.5,0.0)
    
    if args.train.upper() == 'TRAIN':
      for e in range(EPISODES):
        # Episode initialization
        done = False
        score = 0
        loss_list = []
        state = env.reset()
        while not done:
          # if e%100 == 0: env.render()
          # Interact with env.
          action = agent.get_action(state)
          next_state, reward, done, info = env.step(action)
          agent.remember(state, action, reward, next_state, done, goal)
          loss = agent.train_model()
          agent.update_model(done)
          state = next_state
          # 
          score += reward
          loss_list.append(loss)
          global_steps += 1
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
            scores_avg.append(score_avg)
            scores_raw.append(score)
            epsilons.append(agent.epsilon)
            losses.append(np.mean(loss_list))
            if e % save_freq == 0:
              plt.clf()
              plt.subplot(311)
              plt.plot(scores_avg, 'b')
              plt.plot(scores_raw, 'b', alpha=0.8, linewidth=0.5)
              plt.xlabel('episode'); plt.ylabel('average score'); plt.grid()
              plt.title(FILENAME)
              plt.subplot(312)
              plt.plot(epsilons, 'b')
              plt.xlabel('episode'); plt.ylabel('epsilon'); plt.grid()
              plt.subplot(313)
              plt.plot(losses, 'b')
              plt.xlabel('episode'); plt.ylabel('losses') ;plt.grid()
              plt.savefig(workspace_path + "\\result\\img\\" + FILENAME + "_TF.jpg", dpi=100)
            # 이동 평균이 0 이상일 때 종료
            if score_avg > END_SCORE:
              agent.save_model(workspace_path + "\\result\\save_model\\")
              plt.clf()
              plt.subplot(311)
              plt.plot(scores_avg, 'b')
              plt.plot(scores_raw, 'b', alpha=0.8, linewidth=0.5)
              plt.xlabel('episode'); plt.ylabel('average score'); plt.grid()
              plt.title(FILENAME)
              plt.subplot(312)
              plt.plot(epsilons, 'b')
              plt.xlabel('episode'); plt.ylabel('epsilon'); plt.grid()
              plt.subplot(313)
              plt.plot(losses, 'b')
              plt.xlabel('episode'); plt.ylabel('losses') ;plt.grid()
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
      for e in range(5):
        # Episode initialization
        done = False; state = env.reset(); score = 0
        while not done:
          env.render()
          action = agent.choose_action(state)
          next_state, reward, done, info = env.step(action)
          score += reward
          state = next_state
          if done:
            score_avg = 0.9 * score_avg + 0.1 * score if score_avg != 0 else score
            print("episode: {0:3d} | score avg: {1:3.2f} |"
              .format(e, score_avg))
    else:
      print(args.train)
    env.close()
    print("End")