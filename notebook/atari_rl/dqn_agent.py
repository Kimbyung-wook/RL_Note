import gym
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input, ReLU, Conv2D, Flatten
from tensorflow.keras.optimizers import Adam
from ER import ReplayMemory
from PER import ProportionalPrioritizedMemory
from q_network import get_q_network

class DQNAgent():
  def __init__(self, env, cfg):
    self.env_cfg    = cfg['ENV']
    self.rl_cfg     = cfg['RL']
    self.er_cfg     = cfg['RL']['ER']
    self.er_type    = self.er_cfg["ALGORITHM"].upper()
    self.img_size   = self.env_cfg['IMG_SIZE']
    self.state_size = self.env_cfg['IMG_SIZE']
    self.action_size= env.action_space.n

    # Hyper-parameters for learning
    self.discount_factor = 0.99
    self.learning_rate  = 0.005
    self.epsilon        = 1.0
    self.epsilon_decay  = 0.999
    self.epsilon_min    = 0.01
    self.tau            = 0.005
    self.start_to_train = self.er_cfg["TRAIN_START"]
    self.batch_size     = self.er_cfg["BATCH_SIZE"]
    self.buffer_size    = self.er_cfg["MEMORY_SIZE"]
    self.update_freq    = self.rl_cfg['UPDATE_FREQ']
    self.train_freq     = self.rl_cfg['TRAIN_FREQ']

    # DQN Architecture
    self.model        = get_q_network(self.state_size, self.action_size)
    self.target_model = get_q_network(self.state_size, self.action_size)
    self.optimizer    = Adam(learning_rate=self.learning_rate, clipnorm=1.0)
    self.model.summary()

    # Experience Replay
    if self.er_type == "ER":
      self.memory = ReplayMemory(capacity=self.buffer_size)
    elif self.er_type == "PER":
      self.memory = ProportionalPrioritizedMemory(capacity=self.buffer_size)

    # Miscellaneous
    self.show_media_info = False
    self.steps = 0
    
  def get_actions(self, state):
    self.steps += 1
    # Exploration and Exploitation
    if ((np.random.rand() <= self.epsilon) or ()):
      return random.randrange(self.action_size)
    else:
      state = tf.convert_to_tensor([state], dtype=tf.float32)
      return np.argmax(self.model(state))

  def remember(self, state, action, reward, next_state, done):
    state       = np.array(state,       dtype=np.float32)
    action      = np.array([action])
    reward      = np.array([reward],    dtype=np.float32)
    done        = np.array([done],      dtype=np.float32)
    next_state  = np.array(next_state,  dtype=np.float32)
    transition  = (state, action, reward, next_state, done)
    self.memory.append(transition)
    return

  def train(self):
    if self.steps < self.start_to_train:
      return 0.0
    # Sampling from the memory
    if self.steps % self.train_freq == 0:
      return 0.0
    if self.er_type == "ER":
      mini_batch = self.memory.sample(self.batch_size)
    elif self.er_type == "PER":
      mini_batch, idxs, is_weights = self.memory.sample(self.batch_size)

    states      = tf.convert_to_tensor(np.array([sample[0] for sample in mini_batch]))
    actions     = tf.convert_to_tensor(np.array([sample[1][0] for sample in mini_batch]))
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

    model_params = self.model.trainable_variables
    with tf.GradientTape() as tape:
      # get q value
      q = self.model(states)
      one_hot_action = tf.one_hot(actions, self.action_size)
      q = tf.reduce_sum(one_hot_action * q, axis=1)
      q = tf.expand_dims(q,axis=1)
      # Target q and maximum target q
      target_q = tf.stop_gradient(self.target_model(next_states))
      max_q = tf.reduce_max(target_q,axis=1)
      max_q = tf.expand_dims(max_q,axis=1)
      
      targets = rewards + (1 - dones) * self.discount_factor * max_q
      td_error = targets - q
      if self.er_type == "PER":
        loss = tf.reduce_mean(is_weights * tf.square(targets - q))
      else:
        loss = tf.reduce_mean(tf.square(targets - q))
        
    grads = tape.gradient(loss, model_params)
    self.optimizer.apply_gradients(zip(grads, model_params))

    if self.er_type == "PER":
      sample_importance = td_error.numpy()
      for i in range(self.batch_size):
        self.memory.update(idxs[i], sample_importance[i])

    return loss

  def update_target_net(self):
    if self.steps % self.update_freq == 0:
      self.target_model.set_weights(self.model.get_weights())
    return

  def load_model(self,at):
    self.model.load_weights( at + self.filename + "_TF")
    self.target_model.load_weights(at + self.filename + "_TF")
    return

  def save_model(self,at):
    self.model.save_weights( at + self.filename + "_TF", save_format="tf")
    self.target_model.save_weights(at + self.filename + "_TF", save_format="tf")
    return