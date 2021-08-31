# Find RL_Note path and append sys path
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from dqn_agent import DQNAgent
from env_config import env_configs
from gym_wrapper import GymWrapper
from utils import ImageFeaturization

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(\
        gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*3)])
  except RuntimeError as e:
    print(e)
lists = (
            # (2,'ER'),\
            # (3,'ER'),\
            # (4,'ER'),\
            # (5,'ER'),\
            # (6,'ER'),\
            # (8,'ER'),\
            (2,'PER'),\
            (3,'PER'),\
            (4,'PER'),\
            (5,'PER'),\
            (6,'PER'),\
            (8,'PER'),\
        )
print('Batch list : ',lists)

scores_avg, scores_raw, episodes, losses, epsilons = [], [], [], [], []
def save_statistics(filename):
    # View data
    plt.clf()
    plt.subplot(311)
    plt.plot(scores_avg, 'b')
    plt.plot(scores_raw, 'b', alpha=0.8, linewidth=0.5)
    plt.xlabel('Episodes'); plt.ylabel('average score'); plt.grid()
    plt.title(filename)
    plt.subplot(312)
    plt.plot(epsilons, 'b')
    plt.xlabel('Episodes'); plt.ylabel('epsilon'); plt.grid()
    plt.subplot(313)
    plt.plot(losses, 'b')
    plt.xlabel('Episodes'); plt.ylabel('losses') ;plt.grid()
    plt.savefig(filename + "_TF.jpg", dpi=100)

if __name__ == "__main__":
    for item in lists:
        STATE_TYPE = 'IMG'
        # STATE_TYPE = 'MLP'
        cfg = {\
                "ENV":{
                    "NAME":"CartPole-v1",
                    # "IMG_SIZE":(240,160,4),
                    "IMG_SIZE":(120,40,item[0]),
                    "IMG_CROP":((150,350),(0,-1)),
                    "STATE_TYPE":STATE_TYPE,
                    "IMG_TYPE":"GRAY",
                    # "IMG_TYPE":"RGB",
                    },
                "RL":{
                    "ALGORITHM":'DQN',
                    "STATE_TYPE":STATE_TYPE,
                    # 'TYPE':('',),
                    'TYPE':('DUELING',),
                    "NETWORK":{
                        "LAYER":[128,128],
                    },
                    "ER":
                        {
                            "ALGORITHM":item[1],
                        },
                    "BATCH_SIZE":256,
                    "TRAIN_START":2000,
                    "MEMORY_SIZE":50000,
                    },
                "ADD_NAME":(STATE_TYPE,str(item[0]),),
                # "ADD_NAME":STATE_TYPE,
                }
        env_config = env_configs[cfg["ENV"]["NAME"]]
        RL_NAME = cfg["RL"]["ALGORITHM"]
        for item in cfg["RL"]['TYPE']:
            RL_NAME = RL_NAME + '_' + item
        FILENAME = cfg["ENV"]["NAME"] + '_' + RL_NAME + '_' + cfg["RL"]["ER"]["ALGORITHM"]
        if cfg['RL']["ER"]["ALGORITHM"] == "HER":
            FILENAME = FILENAME + '_' + cfg["ER"]["STRATEGY"]
        for item in cfg["ADD_NAME"]:
            FILENAME  = FILENAME + '_' + item
        EPISODES = env_config["EPISODES"]
        END_SCORE = env_config["END_SCORE"]

        env = GymWrapper(cfg=cfg['ENV'])

        if cfg["RL"]["ALGORITHM"] == "DQN":
            agent = DQNAgent(env, cfg)
        # elif cfg["RL"]["ALGORITHM"] == "MDQN":
        #     agent = MDQNAgent(env, cfg)
        image_featurization = ImageFeaturization(data_format = 'last', img_size=cfg['ENV']['IMG_SIZE'])
        plt.clf()
        figure = plt.gcf()
        figure.set_size_inches(8,6)

        save_freq = 10; global_steps = 0
        score_avg = 0
        end = False
        show_media_info = True
        goal = (0.5,0.0)
        is_enough = True
        
        for e in range(EPISODES):
            # Episode initialization
            done = False
            score = 0
            loss_list = []
            state = env.reset()
            if cfg['ENV']['STATE_TYPE'] == 'IMG':
                state, is_enough = image_featurization(state)
            while not done:
                # env.render()
                # Interact with env.
                if is_enough == False:
                    action = random.randrange(env.env.action_space.n)
                else:
                    action = agent.get_action(state)
                next_state, reward, done, info = env.step(action)
                if cfg['ENV']['STATE_TYPE'] == 'IMG':
                    next_state, is_enough = image_featurization(next_state)
                agent.remember(state, action, reward, next_state, done, goal)
                loss = agent.train_model()
                agent.update_model(done)
                state = next_state
                # 
                score += reward
                loss_list.append(loss)
                global_steps+=1
                # break
                if show_media_info:
                    print("-------------- Variable shapes --------------")
                    print("State Shape : ", np.shape(state))
                    print("Action Shape : ", np.shape(action))
                    print("Reward Shape : ", np.shape(reward))
                    print("done Shape : ", np.shape(done))
                    print("---------------------------------------------")
                    if cfg['ENV']['STATE_TYPE'] == "IMG":
                        # print(np.shape(state))
                        # print(np.shape(state[:,:,0]))
                        # plt.imshow(np.squeeze(state,axis=2),cmap='gray')
                        plt.imshow(state[:,:,0],cmap='gray')
                        # plt.imshow(state)
                    show_media_info = False
                if done == True:
                    score_avg = 0.9 * score_avg + 0.1 * score if score_avg != 0 else score
                    print("episode: {0:3d} | score avg: {1:3.2f} | mem size {2:6d} |"
                        .format(e, score_avg, len(agent.memory)))

                    # episodes.append(e)
                    scores_avg.append(score_avg)
                    scores_raw.append(score)
                    losses.append(np.mean(loss_list))
                    epsilons.append(agent.epsilon)
                    if e % save_freq == 0:
                        save_statistics(FILENAME)
                    # 이동 평균이 0 이상일 때 종료
                    if score_avg > END_SCORE:
                        agent.save_model("")
                        save_statistics(FILENAME)
                        end = True
                        break
            if end == True:
                env.close()
                print("End")
                break