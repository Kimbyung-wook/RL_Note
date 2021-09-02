import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from pys.utils.ER import ReplayMemory
from pys.utils.PER import ProportionalPrioritizedMemory
from pys.utils.HER import HindsightMemory
from pys.model.q_network import get_mlp_network

class DQNAgent:
    def __init__(self, env:object, cfg:dict):
        self.state_size = env.observation_space.shape[0]
        self.action_size= env.action_space.n
        self.env_name   = cfg["ENV"]
        self.rl_type    = cfg["RL"]['TYPE']
        self.er_type    = cfg["ER"]["ALGORITHM"].upper()
        rl_name = cfg["RL"]["ALGORITHM"]
        for item in cfg['RL']['TYPE']:
            rl_name = rl_name + '_' + item
        self.filename   = cfg["ENV"] + '_' + rl_name + '_' + cfg["ER"]["ALGORITHM"]
        if cfg["ER"]["ALGORITHM"] == "HER":
            self.filename = self.filename + '_' + cfg["ER"]["STRATEGY"]
        for item in cfg["ADD_NAME"]:
            self.filename = self.filename + '_' + item

        # Experience Replay
        self.batch_size     = cfg["BATCH_SIZE"]
        self.train_start    = cfg["TRAIN_START"]
        self.buffer_size    = cfg["MEMORY_SIZE"]
        if self.er_type == "ER":
            self.memory = ReplayMemory(capacity=self.buffer_size)
        elif self.er_type == "PER":
            self.memory = ProportionalPrioritizedMemory(capacity=self.buffer_size)
        elif self.er_type == "HER":
            self.memory = HindsightMemory(\
                capacity        = self.buffer_size,\
                replay_n        = cfg["ER"]["REPLAY_N"],\
                replay_strategy = cfg["ER"]["STRATEGY"],\
                reward_func     = cfg["ER"]["REWARD_FUNC"],\
                done_func       = cfg["ER"]["DONE_FUNC"])

        # Hyper-parameters for learning
        self.discount_factor = 0.99
        self.learning_rate  = 0.001
        self.epsilon        = 1.0
        self.epsilon_decay  = 0.999
        self.epsilon_min    = 0.01
        self.tau            = 0.005
        
        # Neural Network Architecture
        self.model          = get_mlp_network(self.state_size, self.action_size, cfg["RL"])
        self.target_model   = get_mlp_network(self.state_size, self.action_size, cfg["RL"])
        self.optimizer      = Adam(learning_rate=self.learning_rate)
        self.model.summary()
        self.hard_update_target_model()
        
        # Miscellaneous
        self.show_media_info = False
        self.steps = 0
        self.update_period = 100
        # self.interaction_period = 1

        print(self.filename)
        print('States {0}, Actions {1}'.format(self.state_size, self.action_size))

    def get_action(self,state):
        self.steps += 1
        # Exploration and Exploitation
        if (np.random.rand() <= self.epsilon):
            return random.randrange(self.action_size)
        else:
            state = tf.convert_to_tensor([state], dtype=tf.float32)
            return np.argmax(self.model(state))
        
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

        states      = tf.convert_to_tensor(np.array([sample[0]    for sample in mini_batch]))
        actions     = tf.convert_to_tensor(np.array([sample[1][0] for sample in mini_batch]))
        rewards     = tf.convert_to_tensor(np.array([sample[2]    for sample in mini_batch]))
        next_states = tf.convert_to_tensor(np.array([sample[3]    for sample in mini_batch]))
        dones       = tf.convert_to_tensor(np.array([sample[4]    for sample in mini_batch]))
        
        if self.show_media_info == False:
            self.show_media_info = True
            print('Start to train, check batch shapes')
            print('**** shape of mini_batch',   np.shape(mini_batch),   type(mini_batch))
            print('**** shape of states',       np.shape(states),       type(states))
            print('**** shape of actions',      np.shape(actions),      type(actions))
            print('**** shape of rewards',      np.shape(rewards),      type(rewards))
            print('**** shape of next_states',  np.shape(next_states),  type(next_states))
            print('**** shape of dones',        np.shape(dones),        type(dones))
            if self.er_type == "HER":
                goals = tf.convert_to_tensor(np.array([sample[5] for sample in mini_batch]))
                print('**** shape of goals',    np.shape(goals),        type(goals))

        model_params = self.model.trainable_variables
        with tf.GradientTape() as tape:
            # get q value
            q = self.model(states)
            one_hot_action = tf.one_hot(actions, self.action_size)
            q = tf.reduce_sum(one_hot_action * q, axis=1)
            q = tf.expand_dims(q,axis=1)
            # Target q and maximum target q
            target_q = tf.stop_gradient(self.target_model(next_states))
            if "DOUBLE" in self.rl_type: # Double Estimator
                new_actions = np.argmax(self.model(next_states))
                one_hot_action = tf.one_hot(new_actions, self.action_size)
                max_q = tf.reduce_sum(one_hot_action * target_q, axis=1)
            else: # Single Estimator
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

    def update_model(self,done=False):
        # if done == True:
        if self.steps % self.update_period != 0:
            self.hard_update_target_model()
        return

    def load_model(self,at):
        self.model.load_weights( at + self.filename + "_TF")
        self.target_model.load_weights(at + self.filename + "_TF")
        return

    def save_model(self,at):
        self.model.save_weights( at + self.filename + "_TF", save_format="tf")
        self.target_model.save_weights(at + self.filename + "_TF", save_format="tf")
        return
        
    def hard_update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def soft_update_target_model(self):
        tau = self.tau
        for (net, target_net) in zip(   self.model.trainable_variables,
                                        self.target_model.trainable_variables):
            target_net.assign(tau * net + (1.0 - tau) * target_net)