import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input

from pys.utils.ou_noise import OUActionNoise
from pys.utils.memory import ReplayMemory
from pys.utils.prioritized_memory import ProportionalPrioritizedMemory
from pys.utils.hindsight_memory import HindsightMemory
from pys.model.actor_critic_deterministic_continuous import Actor, Critic

class TD3Agent:
    def __init__(self, env:object, cfg:dict):
        self.state_size = env.observation_space.shape[0]
        self.action_size= env.action_space.shape[0]
        self.action_min = env.action_space.low[0]
        self.action_max = env.action_space.high[0]
        self.env_name   = cfg["ENV"]
        self.rl_type    = "TD3"
        self.er_type    = cfg["ER"]["ALGORITHM"].upper()
        self.filename   = cfg["ENV"] + '_' + cfg["RL"]["ALGORITHM"] + '_' + cfg["ER"]["ALGORITHM"]

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
            self.filename = cfg["ENV"] + '_' + cfg["RL"]["ALGORITHM"] + '_' + cfg["ER"]["ALGORITHM"] + '_' + cfg["ER"]["STRATEGY"]
        
        # Hyper params for learning
        self.discount_factor = 0.99
        self.actor_learning_rate  = 0.001
        self.critic_learning_rate = 0.002
        self.tau = 0.005

        # Networks
        self.critic1        = Critic(self.state_size, self.action_size)
        self.critic2        = Critic(self.state_size, self.action_size)
        self.target_critic1 = Critic(self.state_size, self.action_size)
        self.target_critic2 = Critic(self.state_size, self.action_size)
        self.actor          = Actor(self.state_size, self.action_size, self.action_min, self.action_max)
        self.target_actor   = Actor(self.state_size, self.action_size, self.action_min, self.action_max)
        self.critic1_optimizer   = tf.keras.optimizers.Adam(lr=self.critic_learning_rate)
        self.critic2_optimizer   = tf.keras.optimizers.Adam(lr=self.critic_learning_rate)
        self.actor_optimizer    = tf.keras.optimizers.Adam(lr=self.actor_learning_rate)

        self.actor.build(input_shape=(None, self.state_size))
        self.target_actor.build(input_shape=(None, self.state_size))
        state_in = Input(shape=(self.state_size,),dtype=tf.float32)
        action_in = Input(shape=(self.action_size,),dtype=tf.float32)
        self.actor(state_in)
        self.target_actor(state_in)
        self.critic1([state_in, action_in])
        self.critic2([state_in, action_in])
        self.target_critic1([state_in, action_in])
        self.target_critic2([state_in, action_in])
        # self.actor.summary()
        # self.critic1.summary()
        # self.critic2.summary()
        self.hard_update_target_model()

        # Noise
        self.noise_std_dev = 0.2
        self.noise_mean = 0.0

        # Miscellaneous
        self.update_freq = 1
        self.train_idx = 0
        self.show_media_info = False

        print(self.filename)
        print('States {0}, Actions {1}'.format(self.state_size, self.action_size))
        for i in range(self.action_size):
            print(i+1,'th Action space {0:.2f} ~ {1:.2f}'.format(env.action_space.low[i], env.action_space.high[i]))

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

    def hard_update_target_model(self):
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic1.set_weights(self.critic1.get_weights())
        self.target_critic2.set_weights(self.critic2.get_weights())

    def soft_update_target_model(self):
        tau = self.tau
        for (net, target_net) in zip(   self.actor.trainable_variables,
                                        self.target_actor.trainable_variables):
            target_net.assign(tau * net + (1.0 - tau) * target_net)
        for (net, target_net) in zip(   self.critic1.trainable_variables,
                                        self.target_critic1.trainable_variables):
            target_net.assign(tau * net + (1.0 - tau) * target_net)
        for (net, target_net) in zip(   self.critic2.trainable_variables,
                                        self.target_critic2.trainable_variables):
            target_net.assign(tau * net + (1.0 - tau) * target_net)

    def get_action(self,state):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        action = self.actor(state)
        noise = np.random.randn(self.action_size)*self.noise_std_dev + self.noise_mean
        # Exploration and Exploitation
        return np.clip(action.numpy()[0]+noise,self.action_min,self.action_max)

    def train_model(self):
        # Train from Experience Replay
        # Training Condition - Memory Size
        if len(self.memory) < self.train_start:
            return 0.0, 0.0
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
        target_q1 = self.target_critic1([next_states,target_actions],training=True)
        target_q2 = self.target_critic2([next_states,target_actions],training=True)
        target_q_min = tf.minimum(target_q1, target_q2) # Clipping Double Q
        target_value = rewards + (1.0 - dones) * self.discount_factor * target_q_min

        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            q1 = self.critic1([states, actions], training=True)
            q2 = self.critic2([states, actions], training=True)
            td_error = (tf.abs(target_value - q1) + tf.abs(target_value - q2))/2.0
            if self.er_type == "ER" or self.er_type == "HER":
                critic1_loss = tf.math.reduce_mean(tf.math.square(target_value - q1))
                critic2_loss = tf.math.reduce_mean(tf.math.square(target_value - q2))
            elif self.er_type == "PER":
                critic1_loss = tf.math.reduce_mean(is_weights * tf.math.square(target_value - q1))
                critic2_loss = tf.math.reduce_mean(is_weights * tf.math.square(target_value - q2))
        critic1_params = self.critic1.trainable_variables
        critic2_params = self.critic2.trainable_variables
        critic1_grads = tape1.gradient(critic1_loss, critic1_params)
        critic2_grads = tape2.gradient(critic2_loss, critic2_params)
        self.critic1_optimizer.apply_gradients(zip(critic1_grads, critic1_params))
        self.critic2_optimizer.apply_gradients(zip(critic2_grads, critic2_params))

        actor_loss_out = 0.0
        self.train_idx = self.train_idx + 1
        if self.train_idx % self.update_freq == 0:
            with tf.GradientTape() as tape:
                new_actions = self.actor(states,training=True)
                new_q = self.critic1([states, new_actions],training=True)
                actor_loss = -tf.reduce_mean(new_q)
            actor_loss_out = actor_loss.numpy()
            actor_params = self.actor.trainable_variables
            actor_grads = tape.gradient(actor_loss, actor_params)
            self.actor_optimizer.apply_gradients(zip(actor_grads, actor_params))
            self.soft_update_target_model()

        if self.er_type == "PER":
            sample_importance = td_error.numpy()
            for i in range(self.batch_size):
                self.memory.update(idxs[i], sample_importance[i])

        critic_loss_out = 0.5*(critic1_loss.numpy() + critic2_loss.numpy())
        return critic_loss_out, actor_loss_out
        
    def load_model(self,at):
        self.actor.load_weights(  at + self.filename + "_TF_actor")
        self.critic1.load_weights(at + self.filename + "_TF_critic1")
        self.critic2.load_weights(at + self.filename + "_TF_critic2")
        return

    def save_model(self,at):
        self.actor.save_weights(  at + self.filename + "_TF_actor", save_format="tf")
        self.critic1.save_weights(at + self.filename + "_TF_critic1", save_format="tf")
        self.critic2.save_weights(at + self.filename + "_TF_critic2", save_format="tf")
        return