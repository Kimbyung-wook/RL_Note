
# Find RL_Note path and append sys path
import os
import sys

cwd = os.getcwd()
pos = cwd.find('RL_Note')
root_path = cwd[0:pos] + 'RL_Note'
sys.path.append(root_path)
print(root_path)
workspace_path = root_path + "\\pys"

import gym
import random
import numpy as np
import matplotlib.pyplot as plt
from env_config  import env_configs
from pys.agent.ddpg_agent   import DDPGAgent
from pys.agent.td3_agent    import TD3Agent
from pys.agent.sac_agent    import SACAgent
from pys.gyms.functions import lunarlandercontinuous_done as done_function
from pys.gyms.functions import lunarlandercontinuous_reward as reward_function

# lists = (   ('SAC','ER'),('SAC','PER'),('SAC','HER'),\
#             ('TD3','ER'),('TD3','PER'),('TD3','HER'),\
#             ('DDPG','ER'),('DDPG','PER'),('DDPG','HER'),\
#         )
lists = (
            ('TD3','PER'),\
            ('DDPG','ER'),('DDPG','PER'),\
        )
        
if __name__ == "__main__":
    for item in lists:
        cfg = {\
                # "ENV":"Pendulum-v0",\
                "ENV":"LunarLanderContinuous-v2",\
                "RL":{
                    "ALGORITHM":item[0],\
                },\
                "ER":\
                    {
                        "ALGORITHM":item[1],\
                        "REPLAY_N":8,\
                        "STRATEGY":"FINAL",\
                        "REWARD_FUNC":reward_function,\
                        "DONE_FUNC":done_function,\
                    },\
                "BATCH_SIZE":32,\
                "TRAIN_START":2000,\
                "MEMORY_SIZE":50000,\
                }
        env_config = env_configs[cfg["ENV"]]
        if cfg["ER"]["ALGORITHM"] == "HER":
            FILENAME = cfg["ENV"] + '_' + cfg["RL"]["ALGORITHM"] + '_' + cfg["ER"]["ALGORITHM"] + '_' + cfg["HER"]["STRATEGY"]
        else:
            FILENAME = cfg["ENV"] + '_' + cfg["RL"]["ALGORITHM"] + '_' + cfg["ER"]["ALGORITHM"]
        EPISODES = env_config["EPISODES"]
        END_SCORE = env_config["END_SCORE"]
        plt.clf()
        figure = plt.gcf()
        figure.set_size_inches(8,6)
        env = gym.make(cfg["ENV"])
        if cfg["RL"]["ALGORITHM"] == "DDPG":
            agent = DDPGAgent(env, cfg)
        elif cfg["RL"]["ALGORITHM"] == "TD3":
            agent = TD3Agent(env, cfg)
        elif cfg["RL"]["ALGORITHM"] == "SAC":
            agent = SACAgent(env, cfg)

        scores_avg, scores_raw, episodes, losses = [], [], [], []
        critic_mean, actor_mean = [], []
        score_avg = 0
        end = False
        show_media_info = True
        goal = np.array([1.0,0.0,0.0])

        for e in range(EPISODES):
            done = False
            score = 0
            state = env.reset()
            critic_losses = []
            actor_losses = []
            while not done:
                # if e%100 == 0:
                #     env.render()
                # Interact with env.
                action = agent.get_action(state)
                next_state, reward, done, info = env.step(action)
                agent.remember(state, action, reward, next_state, done, goal)
                critic_loss, actor_loss = agent.train_model()
                state = next_state
                # 
                score += reward
                critic_losses.append(critic_loss)
                actor_losses.append(actor_loss)
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
                    # View data
                    plt.clf()
                    plt.subplot(311)
                    plt.plot(episodes, scores_avg, 'b')
                    plt.plot(episodes, scores_raw, 'b', alpha=0.8, linewidth=0.5)
                    plt.xlabel('episode'); plt.ylabel('average score'); plt.grid()
                    plt.title(cfg["ENV"] +'_' + cfg["RL"]["ALGORITHM"] +'_' + cfg["ER"])
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
                        end = True
                        break
            if end == True:
                env.close()
                np.save(workspace_path + "\\result\\data\\" + FILENAME + "_TF_epi",  episodes)
                np.save(workspace_path + "\\result\\data\\" + FILENAME + "_TF_scores_avg",scores_avg)
                np.save(workspace_path + "\\result\\data\\" + FILENAME + "_TF_scores_raw",scores_raw)
                np.save(workspace_path + "\\result\\data\\" + FILENAME + "_TF_critic_mean",critic_mean)
                np.save(workspace_path + "\\result\\data\\" + FILENAME + "_TF_actor_mean",actor_mean)
                print("End")
                break