import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input

from pys.utils.ou_noise import OUActionNoise
from pys.utils.memory import ReplayMemory
from pys.utils.prioritized_memory import ProportionalPrioritizedMemory
from pys.utils.hindsight_memory import HindsightMemory
from pys.model.actor_critic_deterministic_continuous import Actor, Critic

class DDPGAgent:
    def __init__(self, env:object, cfg:dict):
        self.state_size = env.observation_space.shape[0]
        self.action_size= env.action_space.shape[0]
        self.action_min = env.action_space.low[0]
        self.action_max = env.action_space.high[0]
        self.env_name   = cfg["ENV"]
        self.rl_type    = "DDPG"
        self.er_type    = cfg["ER"]["ALGORITHM"].upper()
        self.filename = cfg["ENV"] + '_' + cfg["RL"]["ALGORITHM"] + '_' + cfg["ER"]["ALGORITHM"]
        if cfg["ER"]["ALGORITHM"] == "HER":
            self.filename = self.filename + '_' + cfg["ER"]["STRATEGY"]
        self.filename = self.filename + '_' + cfg["ADD_NAME"]

        # Experience Replay
        self.batch_size = cfg["BATCH_SIZE"]
        self.train_start = cfg["TRAIN_START"]
        self.buffer_size = cfg["MEMORY_SIZE"]
        if self.er_type == "ER":
            self.memory = ReplayMemory(capacity=self.buffer_size)
        elif self.er_type == "PER":
            self.memory = ProportionalPrioritizedMemory(capacity=self.buffer_size)
        elif self.er_type == "HER":
            self.memory = HindsightMemory(\
                capacity            = self.buffer_size,\
                replay_n            = cfg["ER"]["REPLAY_N"],\
                replay_strategy     = cfg["ER"]["STRATEGY"],\
                reward_func         = cfg["ER"]["REWARD_FUNC"],\
                done_func           = cfg["ER"]["DONE_FUNC"])

        # Hyper params for learning
        self.discount_factor = 0.99
        self.actor_learning_rate  = 0.001
        self.critic_learning_rate = 0.002
        self.tau = 0.005

        # Networks
        self.critic         = Critic(self.state_size, self.action_size, cfg=cfg['RL']["NETWORK"])
        self.target_critic  = Critic(self.state_size, self.action_size, cfg=cfg['RL']["NETWORK"])
        self.actor          = Actor(self.state_size, self.action_size, self.action_min, self.action_max, cfg=cfg['RL']["NETWORK"])
        self.target_actor   = Actor(self.state_size, self.action_size, self.action_min, self.action_max, cfg=cfg['RL']["NETWORK"])
        self.critic_optimizer   = tf.keras.optimizers.Adam(lr=self.critic_learning_rate)
        self.actor_optimizer    = tf.keras.optimizers.Adam(lr=self.actor_learning_rate)

        self.actor.build(input_shape=(None, self.state_size))
        self.target_actor.build(input_shape=(None, self.state_size))
        state_in = Input((self.state_size,))
        action_in = Input((self.action_size,))
        self.actor(state_in)
        self.target_actor(state_in)
        self.critic([state_in, action_in])
        self.target_critic([state_in, action_in])
        # self.actor.summary()
        # self.critic.summary()
        self.hard_update_target_model()

        # Noise
        self.noise_std_dev = 0.2
        self.ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(self.noise_std_dev) * np.ones(1))

        # Miscellaneous
        self.update_freq = 1
        self.train_idx = 0
        self.show_media_info = False

        print(self.filename)
        print('States {0}, Actions {1}'.format(self.state_size, self.action_size))
        for i in range(self.action_size):
            print(i+1,'th Action space {0:.2f} ~ {1:.2f}'.format(env.action_space.low[i], env.action_space.high[i]))

    def get_action(self,state):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        action = self.actor(state).numpy()[0]
        noise = self.ou_noise()
        # Exploration and Exploitation
        return np.clip(action+noise,self.action_min,self.action_max)

    def remember(self, state, action, reward, next_state, done, goal=None):
        state       = np.array(state,       dtype=np.float32)
        action      = np.array(action,      dtype=np.float32)
        reward      = np.array([reward],    dtype=np.float32)
        done        = np.array([done],      dtype=np.float32)
        next_state  = np.array(next_state,  dtype=np.float32)
        if self.er_type == "HER":
            goal        = np.array(goal,        dtype=np.float32)
            transition  = (state, action, reward, next_state, done, goal)
        else:
            transition  = (state, action, reward, next_state, done)
        self.memory.append(transition)
        return

    def train_model(self):
        # Train from Experience Replay
        # Training Condition - Memory Size
        if len(self.memory) < self.train_start:
            return 0.0,0.0
        self.train_idx = self.train_idx + 1
        # Sampling from the memory
        if self.er_type == "ER" or self.er_type == "HER":
            mini_batch = self.memory.sample(self.batch_size)
        elif self.er_type == "PER":
            mini_batch, idxs, is_weights = self.memory.sample(self.batch_size)

        states      = tf.convert_to_tensor(np.array([sample[0] for sample in mini_batch]))
        actions     = tf.convert_to_tensor(np.array([sample[1] for sample in mini_batch]))
        rewards     = tf.convert_to_tensor(np.array([sample[2] for sample in mini_batch]))
        next_states = tf.convert_to_tensor(np.array([sample[3] for sample in mini_batch]))
        dones       = tf.convert_to_tensor(np.array([sample[4] for sample in mini_batch]))
        
        if self.show_media_info == False:
            self.show_media_info = True
            print('Start to train, check batch shapes')
            print('**** shape of mini_batch', np.shape(mini_batch),type(mini_batch))
            print('**** shape of states', np.shape(states),type(states))
            print('**** shape of actions', np.shape(actions),type(actions))
            print('**** shape of rewards', np.shape(rewards),type(rewards))
            print('**** shape of next_states', np.shape(next_states),type(next_states))
            print('**** shape of dones', np.shape(dones),type(dones))
            if self.er_type == "HER":
                goals = tf.convert_to_tensor(np.array([sample[5] for sample in mini_batch]))
                print('**** shape of goals', np.shape(goals),type(goals))

        # Update critic
        target_actions = self.target_actor(next_states,training=True)
        target_q = self.target_critic([next_states,target_actions],training=True)
        target_value = rewards + (1 - dones) * self.discount_factor * target_q

        with tf.GradientTape() as tape:
            q = self.critic([states, actions],training=True)
            td_error = tf.abs(target_value - q)
            if self.er_type == "ER" or self.er_type == "HER":
                critic_loss = tf.math.reduce_mean(tf.math.square(target_value - q))
            elif self.er_type == "PER":
                critic_loss = tf.math.reduce_mean(is_weights * tf.math.square(target_value - q))
        critic_loss_out = critic_loss.numpy()
        critic_params = self.critic.trainable_variables
        critic_grads = tape.gradient(critic_loss, critic_params)
        self.critic_optimizer.apply_gradients(zip(critic_grads, critic_params))

        # Update critic
        with tf.GradientTape() as tape:
            new_actions = self.actor(states,training=True)
            new_q = self.critic([states, new_actions],training=True)
            actor_loss = -tf.reduce_mean(new_q)
        actor_loss_out = actor_loss.numpy()
        actor_params = self.actor.trainable_variables
        actor_grads = tape.gradient(actor_loss, actor_params)
        self.actor_optimizer.apply_gradients(zip(actor_grads, actor_params))
        
        if self.er_type == "PER":
            sample_importance = td_error.numpy()
            for i in range(self.batch_size):
                self.memory.update(idxs[i], sample_importance[i])

        return critic_loss_out, actor_loss_out

    def update_model(self,done=False):
        if self.train_idx % self.update_freq == 0:
            self.soft_update_target_model()
        return

    def load_model(self,at):
        self.actor.load_weights( at + self.filename + "_TF_actor")
        self.critic.load_weights(at + self.filename + "_TF_critic")
        return

    def save_model(self,at):
        self.actor.save_weights( at + self.filename + "_TF_actor", save_format="tf")
        self.critic.save_weights(at + self.filename + "_TF_critic", save_format="tf")
        return

    def hard_update_target_model(self):
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())

    def soft_update_target_model(self):
        tau = self.tau
        for (net, target_net) in zip(   self.actor.trainable_variables,
                                        self.target_actor.trainable_variables):
            target_net.assign(tau * net + (1.0 - tau) * target_net)
        for (net, target_net) in zip(   self.critic.trainable_variables,
                                        self.target_critic.trainable_variables):
            target_net.assign(tau * net + (1.0 - tau) * target_net)
