import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input

from pys.utils.memory import ReplayMemory
from pys.utils.prioritized_memory import ProportionalPrioritizedMemory
from pys.utils.hindsight_memory import HindsightMemory
from pys.model.q_network import QNetwork

class MDQNAgent:
    def __init__(self, env:object, cfg:dict):
        self.state_size = env.observation_space.shape[0]
        self.action_size= env.action_space.n
        self.env_name   = cfg["ENV"]
        self.rl_type    = "MDQN"
        self.er_type    = cfg["ER"].upper()
        self.filename   = cfg["ENV"] + '_' + cfg["RL"]["ALGORITHM"] + '_' + cfg["ER"]
        
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
                replay_n            = cfg["HER"]["REPLAY_N"],\
                replay_strategy     = cfg["HER"]["STRATEGY"],\
                reward_func         = cfg["HER"]["REWARD_FUNC"],\
                done_func           = cfg["HER"]["DONE_FUNC"])
            self.filename = cfg["ENV"] + '_' + cfg["RL"]["ALGORITHM"] + '_' + cfg["ER"] + '_' + cfg["HER"]["STRATEGY"]

        # Hyper-parameters for learning
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.entropy_tau = 0.03
        self.alpha = 0.9        
        self.lo = -1
        
        # Neural Network Architecture
        self.model        = QNetwork(self.state_size, self.action_size, cfg)
        self.target_model = QNetwork(self.state_size, self.action_size, cfg)
        self.optimizer = tf.keras.optimizers.Adam(lr=self.learning_rate)
        self.hard_update_target_model()
        
        # Miscellaneous
        self.show_media_info = False
        self.steps = 0
        self.update_period = 2
        self.interaction_period = 1
        self.is_done = False

        print(self.filename)
        print('States {0}, Actions {1}'.format(self.state_size, self.action_size))
        
    def remember(self, state, action, reward, next_state, done, goal=None):
        state       = np.array(state,       dtype=np.float32)
        action      = np.array([action])
        reward      = np.array([reward],    dtype=np.float32)
        done        = np.array([done],      dtype=np.float32)
        next_state  = np.array(next_state,  dtype=np.float32)
        self.is_done = done
        if self.er_type == "HER":
            goal        = np.array(goal,        dtype=np.float32)
            transition  = (state, action, reward, next_state, done, goal)
        else:
            transition  = (state, action, reward, next_state, done)
        self.memory.append(transition)
        return
        
    def hard_update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def soft_update_target_model(self):
        tau = self.tau
        for (net, target_net) in zip(   self.model.trainable_variables,
                                        self.target_model.trainable_variables):
            target_net.assign(tau * net + (1.0 - tau) * target_net)

    def get_action(self,state):
        self.steps += 1
        # Exploration and Exploitation
        if (np.random.rand() <= self.epsilon):
            return random.randrange(self.action_size)
        else:
            state = tf.convert_to_tensor([state], dtype=tf.float32)
            return np.argmax(self.model.predict(state))
        
    def train_model(self):
        # Train from Experience Replay
        # Training Condition - Memory Size
        if len(self.memory) < self.train_start:
            return 0.0
        # Interaction period
        if self.steps % self.interaction_period != 0:
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

        model_params = self.model.trainable_variables
        with tf.GradientTape() as tape:
            # get q value
            q = self.model(states)
            one_hot_action = tf.one_hot(actions, self.action_size)
            q = tf.reduce_sum(one_hot_action * q, axis=1)
            # munchausen reward
            target_q = tf.stop_gradient(self.target_model(states))
            target_v = tf.reduce_max(target_q,axis=1)
            target_v = tf.expand_dims(target_v,axis=1)

            logsum = tf.math.reduce_logsumexp(((target_q - target_v)/self.entropy_tau),axis=1)
            logsum = tf.expand_dims(logsum,axis=1)
            log_pi = target_q - target_v - self.entropy_tau * logsum
            one_hot_action = tf.one_hot(actions, self.action_size)
            log_pi = tf.reduce_sum(one_hot_action * log_pi, axis=1)
            log_pi = tf.expand_dims(log_pi,axis=1)
            
            term_1st = rewards + self.alpha * self.entropy_tau * tf.clip_by_value(log_pi, clip_value_min=self.lo, clip_value_max=0.0)

            # munchausen target value y
            target_next_q = tf.stop_gradient(self.target_model(next_states))
            target_next_v = tf.reduce_max(target_next_q,axis=1)
            target_next_v = tf.expand_dims(target_next_v,axis=1)
            next_logsum = tf.math.reduce_logsumexp(((target_next_q - target_next_v)/self.entropy_tau),axis=1)
            next_logsum = tf.expand_dims(next_logsum,axis=1)
            target_next_log_pi = target_next_q - target_next_v - self.entropy_tau * next_logsum
            target_pi = tf.math.softmax(target_next_q/self.entropy_tau,1)

            term_2nd = self.discount_factor * \
                tf.reduce_sum(target_pi * (target_next_q - target_next_log_pi) * (1 - dones), axis=1)

            target_y = term_1st + term_2nd
            
            # Compute loss
            loss = tf.reduce_mean(tf.square(target_y - q))
            
        grads = tape.gradient(loss, model_params)
        self.optimizer.apply_gradients(zip(grads, model_params))
        
        if self.er_type == "PER":
            sample_importance = td_error.numpy()
            for i in range(self.batch_size):
                self.memory.update(idxs[i], sample_importance[i])

        return loss

    def update_network(self):
        # if self.steps % self.update_period != 0:
        if self.is_done:
            self.hard_update_target_model()
        return

    def load_model(self,at):
        self.actor.load_weights( at + self.filename + "_TF")
        self.critic.load_weights(at + self.filename + "_TF")
        return

    def save_model(self,at):
        self.actor.save_weights( at + self.filename + "_TF", save_format="tf")
        self.critic.save_weights(at + self.filename + "_TF", save_format="tf")
        return