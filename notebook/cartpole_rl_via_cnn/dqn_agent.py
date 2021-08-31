import random
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input

from er import ReplayMemory
from per import ProportionalPrioritizedMemory
from her import HindsightMemory
from q_network import MLPNetwork
from q_network import CNN3Network as CNNNetwork


class DQNAgent:
    def __init__(self, env:object, cfg:dict):
        self.state_size = env.observation_space.shape
        self.action_size= env.action_space.n
        self.env_cfg    = cfg['ENV']
        self.rl_cfg     = cfg['RL']
        self.er_cfg     = cfg['RL']['ER']
        self.rl_type    = self.rl_cfg['ALGORITHM']
        self.er_type    = self.er_cfg["ALGORITHM"].upper()
        self.filename   = self.env_cfg['NAME'] + '_' + self.rl_cfg["ALGORITHM"] + '_' + self.er_cfg["ALGORITHM"]
        if self.er_cfg["ALGORITHM"] == "HER":
            self.filename = self.filename + '_' + self.er_cfg["STRATEGY"]
        self.filename = self.filename + '_' + cfg["ADD_NAME"]

        print(self.filename)
        print('States {0}, Actions {1}'.format(self.state_size, self.action_size))

        # Experience Replay
        self.batch_size     = self.rl_cfg["BATCH_SIZE"]
        self.train_start    = self.rl_cfg["TRAIN_START"]
        self.buffer_size    = self.rl_cfg["MEMORY_SIZE"]
        if self.er_type == "ER":
            self.memory = ReplayMemory(capacity=self.buffer_size)
        elif self.er_type == "PER":
            self.memory = ProportionalPrioritizedMemory(capacity=self.buffer_size)
        elif self.er_type == "HER":
            self.memory = HindsightMemory(\
                capacity            = self.buffer_size,\
                replay_n            = self.er_cfg["REPLAY_N"],\
                replay_strategy     = self.er_cfg["STRATEGY"],\
                reward_func         = self.er_cfg["REWARD_FUNC"],\
                done_func           = self.er_cfg["DONE_FUNC"])

        # Hyper-parameters for learning
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.tau = 0.005
        
        # Neural Network Architecture
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        if self.env_cfg['STATE_TYPE'] == "MLP":
            print('Neural Network Model is MLP')
            self.q_net        = MLPNetwork(self.state_size, self.action_size, self.rl_cfg["NETWORK"])
            self.target_q_net = MLPNetwork(self.state_size, self.action_size, self.rl_cfg["NETWORK"])
            # self.q_net.build(input_shape=(None,4))
            # self.target_q_net.build(input_shape=(None, 4))
            # state_in = Input((4,))
            # self.q_net(state_in)
            # self.target_q_net(state_in)
        elif self.env_cfg['STATE_TYPE'] == 'IMG':
            print('Neural Network Model is MLP with CNN')
            self.q_net        = CNNNetwork(self.state_size, self.action_size, self.rl_cfg["NETWORK"])
            self.target_q_net = CNNNetwork(self.state_size, self.action_size, self.rl_cfg["NETWORK"])
            self.q_net.summary()
        # self.target_q_net.summary()
        self.hard_update_target_model()
        
        # Miscellaneous
        self.image_size = self.env_cfg["IMG_SIZE"]
        self.image_crop = self.env_cfg['IMG_CROP']
        self.show_media_info = False
        self.steps = 0
        self.update_period = 10
        # self.interaction_period = 1
        
    def remember(self, state, action, reward, next_state, done, goal=None):
        state       = np.array(state,       dtype=np.float32)
        action      = np.array([action])
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
        self.target_q_net.set_weights(self.q_net.get_weights())

    def soft_update_target_model(self):
        tau = self.tau
        for (net, target_net) in zip(   self.q_net.trainable_variables,
                                        self.target_q_net.trainable_variables):
            target_net.assign(tau * net + (1.0 - tau) * target_net)

    def get_action(self,state):
        self.steps += 1
        # Exploration and Exploitation
        if (np.random.rand() <= self.epsilon):
            return random.randrange(self.action_size)
        else:
            state = tf.convert_to_tensor([state], dtype=tf.float32)
            return np.argmax(self.q_net(state))
        
    def train_model(self):
        # Train from Experience Replay
        # Training Condition - Memory Size
        if len(self.memory) < self.train_start:
            return 0.0
        # Decaying Exploration Ratio
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        # Sampling from the memory
        if self.er_type == "ER" or self.er_type == "HER":
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
            if self.er_type == "HER":
                goals = tf.convert_to_tensor(np.array([sample[5] for sample in mini_batch]))
                print('**** shape of goals', np.shape(goals),type(goals))

        model_params = self.q_net.trainable_variables
        with tf.GradientTape() as tape:
            # get q value
            q = self.q_net(states)
            one_hot_action = tf.one_hot(actions, self.action_size)
            q = tf.reduce_sum(one_hot_action * q, axis=1)
            q = tf.expand_dims(q,axis=1)
            # Target q and maximum target q
            target_q = tf.stop_gradient(self.target_q_net(next_states))
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

    def update_network(self,done=False):
        if done == True:
            self.hard_update_target_model()
        # if self.steps % self.update_period != 0:
        #     self.soft_update_target_model()
        # return

    def load_model(self,at):
        self.q_net.load_weights( at + self.filename + "_TF")
        self.target_q_net.load_weights(at + self.filename + "_TF")
        return

    def save_model(self,at):
        self.q_net.save_weights( at + self.filename + "_TF", save_format="tf")
        self.target_q_net.save_weights(at + self.filename + "_TF", save_format="tf")
        return