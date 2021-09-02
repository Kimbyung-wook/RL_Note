# Find RL_Note path and append sys path
import os, sys
cwd = os.getcwd()
dir_name = 'RL_Note'
tmp1 = cwd.lower()
tmp2 = dir_name.lower()
pos = tmp1.find(tmp2)
root_path = cwd[0:pos] + dir_name
sys.path.append(root_path)
print(root_path)
workspace_path = root_path + "\\pys"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import gym
import numpy as np
import matplotlib.pyplot as plt
from env_config  import env_configs
from pys.agent.agent_broker import discrete_agent_broker
from pys.gyms.functions import mountain_car_done as done_function
from pys.gyms.functions import mountain_car_reward as reward_function
from pys.utils.gpu_memory_limiter import gpu_memory_limiter

gpu_memory_limiter(1024)
    
lists = (
    # ('DQN','ER',''),
    ('DQN','PER',''),
    # ('DQN','ER','DUELING',),
    ('DQN','PER','DUELING',),
    # ('MDQN','ER',''),
    # ('MDQN','PER',''),
    # ('MDQN','ER','DUELING',),
    # ('MDQN','PER','DUELING',),
  )
print('Batch list : ',lists)

if __name__ == "__main__":
  for item in lists:
    cfg = {\
      # "ENV":"Pong-v0",\
      # "ENV":"CartPole-v1",\
      "ENV":"LunarLander-v2",\
      # "ENV":"MountainCar-v0",\
      "RL":{
        "ALGORITHM":item[0],\
        "TYPE":(item[2],),
        "NETWORK":{
          "MLP":(
            (128,'relu'),
            (128,'relu'),
          )
        }
      },\
      "ER":{
        "ALGORITHM":item[1],\
        "REPLAY_N":8,\
        "STRATEGY":"EPISODE",\
        "REWARD_FUNC":reward_function,\
        "DONE_FUNC":done_function,\
      },\
      "BATCH_SIZE":128,\
      "TRAIN_START":2000,\
      "MEMORY_SIZE":100000,\
      "ADD_NAME":()
    }
    env_config = env_configs[cfg["ENV"]]
    RL_NAME = cfg["RL"]["ALGORITHM"]
    for item in cfg['RL']['TYPE']:
      RL_NAME = RL_NAME + '_' + item
    FILENAME = cfg["ENV"] + '_' + RL_NAME + '_' + cfg["ER"]["ALGORITHM"]
    if cfg["ER"]["ALGORITHM"] == "HER":
      FILENAME = FILENAME + '_' + cfg["ER"]["STRATEGY"]
    for item in cfg["ADD_NAME"]:
      FILENAME  = FILENAME + '_' + item
    EPISODES  = env_config["EPISODES"]
    END_SCORE = env_config["END_SCORE"]

    # Define Environment
    env = gym.make(cfg["ENV"])
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
        env.close()
        np.save(workspace_path + "\\result\\data\\" + FILENAME + "_TF_scores_avg",scores_avg)
        np.save(workspace_path + "\\result\\data\\" + FILENAME + "_TF_scores_raw",scores_raw)
        np.save(workspace_path + "\\result\\data\\" + FILENAME + "_TF_losses",losses)
        print("End")
        break