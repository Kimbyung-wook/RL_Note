
# Find RL_Note path and append sys path
import os
import sys

cwd = os.getcwd()
pos = cwd.find('RL_Note')
root_path = cwd[0:pos] + 'RL_Note'
sys.path.append(root_path)
print(root_path)
workspace_path = root_path + "\\pys"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import gym
import numpy as np
import matplotlib.pyplot as plt
from env_config  import env_configs
from pys.agent.dqn_agent import DQNAgent
from pys.agent.mdqn_agent import MDQNAgent

# lists = (   ('SAC','ER'),('SAC','PER'),('SAC','HER'),\
#             ('TD3','ER'),('TD3','PER'),('TD3','HER'),\
#             ('DDPG','ER'),('DDPG','PER'),('DDPG','HER'),\
#         )
lists = (
            ('MDQN','PER'),('MDQN','ER'),\
            ('DQN','ER'),('DQN','PER'),\
        )
        
if __name__ == "__main__":
    for item in lists:
        cfg = {\
                # "ENV":"Pong-v0",\
                "ENV":"CartPole-v1",\
                "RL":{
                    "ALGORITHM":item[0],\
                    "NETWORK":{
                        "LAYER":[255,255],\

                    }
                },\
                "ER":
                    {
                        "ALGORITHM":item[1],\
                        "REPLAY_N":8,\
                        "STRATEGY":"EPISODE",\
                        # "REWARD_FUNC":reward_function,\
                        # "DONE_FUNC":done_function,\
                    },\
                "BATCH_SIZE":8,\
                "TRAIN_START":500,\
                "MEMORY_SIZE":20000,\
                }
        env_config = env_configs[cfg["ENV"]]
        if cfg["ER"]["ALGORITHM"] == "HER":
            FILENAME = cfg["ENV"] + '_' + cfg["RL"]["ALGORITHM"] + '_' + cfg["ER"]["ALGORITHM"] + '_' + cfg["HER"]["STRATEGY"]
        else:
            FILENAME = cfg["ENV"] + '_' + cfg["RL"]["ALGORITHM"] + '_' + cfg["ER"]["ALGORITHM"]
        EPISODES = env_config["EPISODES"]
        END_SCORE = env_config["END_SCORE"]

        env = gym.make(cfg["ENV"])
        if cfg["RL"]["ALGORITHM"] == "DQN":
            agent = DQNAgent(env, cfg)
        elif cfg["RL"]["ALGORITHM"] == "A2C":
            agent = A2CAgent(env, cfg)
        elif cfg["RL"]["ALGORITHM"] == "MDQN":
            agent = MDQNAgent(env, cfg)
        
        plt.clf()
        figure = plt.gcf()
        figure.set_size_inches(8,6)

        scores_avg, scores_raw, episodes, losses = [], [], [], []
        epsilons = []
        score_avg = 0
        end = False
        show_media_info = True
        goal = np.array([1.0,0.0,0.0])
        
        for e in range(EPISODES):
            # Episode initialization
            done = False
            score = 0
            loss_list = []
            state = env.reset()
            while not done:
                # env.render()
                # Interact with env.
                action = agent.get_action(state)
                next_state, reward, done, info = env.step(action)
                agent.remember(state, action, reward, next_state, done, goal)
                loss = agent.train_model()
                agent.update_network()
                state = next_state
                # 
                score += reward
                loss_list.append(loss)
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
                    losses.append(np.mean(loss_list))
                    epsilons.append(agent.epsilon)
                    # View data
                    plt.clf()
                    plt.subplot(311)
                    plt.plot(episodes, scores_avg, 'b')
                    plt.plot(episodes, scores_raw, 'b', alpha=0.8, linewidth=0.5)
                    plt.xlabel('episode'); plt.ylabel('average score'); plt.grid()
                    plt.title(cfg["ENV"] +'_' + cfg["RL"]["ALGORITHM"] +'_' + cfg["ER"])
                    plt.subplot(312)
                    plt.plot(episodes, epsilons, 'b')
                    plt.xlabel('episode'); plt.ylabel('epsilon'); plt.grid()
                    plt.subplot(313)
                    plt.plot(episodes, losses, 'b')
                    plt.xlabel('episode'); plt.ylabel('losses') ;plt.grid()
                    plt.savefig(workspace_path + "\\result\\img\\" + FILENAME + "_TF.jpg", dpi=100)

                    # 이동 평균이 0 이상일 때 종료
                    if score_avg > END_SCORE:
                        agent.save_model(workspace_path + "\\result\\save_model\\")
                        end = True
                        break
            if end == True:
                env.close()
                np.save(workspace_path + "\\result\\data\\" + FILENAME + "_TF_epi",  episodes)
                np.save(workspace_path + "\\result\\data\\" + FILENAME + "_TF_scores_avg",scores_avg)
                np.save(workspace_path + "\\result\\data\\" + FILENAME + "_TF_scores_raw",scores_raw)
                np.save(workspace_path + "\\result\\data\\" + FILENAME + "_TF_losses",losses)
                print("End")
                break