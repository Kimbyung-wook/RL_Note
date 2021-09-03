import random, time
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
tfd = tfp.distributions

from pys.utils.ER import ReplayMemory
from pys.utils.PER import ProportionalPrioritizedMemory
from pys.utils.HER import HindsightMemory
from pys.model.network_maker import network_maker1, network_maker2

class SACAgent1:
    def __init__(self, env:object, cfg:dict):
        self.state_size = env.observation_space.shape
        self.action_size= env.action_space.shape
        # self.board      = board
        self.env_name   = cfg["ENV"]['NAME']
        self.rl_type    = cfg["RL"]['TYPE']
        self.er_type    = cfg["ER"]["ALGORITHM"].upper()
        rl_name = cfg["RL"]["ALGORITHM"]
        for item in cfg['RL']['TYPE']:
            rl_name = rl_name + '_' + item
        self.filename   = self.env_name + '_' + rl_name + '_' + self.er_type
        if cfg["ER"]["ALGORITHM"] == "HER":
            self.filename = self.filename + '_' + cfg["ER"]["STRATEGY"]
        for item in cfg["ADD_NAME"]:
            self.filename = self.filename + '_' + item

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
        self.critic_lr  = 0.002
        self.actor_lr   = 0.001
        self.alpha_lr   = 0.001
        self.tau        = 0.005
        self.alpha      = 0.200 # temperature

        # Networks
        cfg['RL']['NETWORK']['ACTOR']['ACTION_TYPE'] = 'STOCHASTIC'
        self.critic1        = network_maker1(self.state_size, self.action_size, cfg=cfg['RL']['NETWORK']['CRITIC'])
        self.critic2        = network_maker1(self.state_size, self.action_size, cfg=cfg['RL']['NETWORK']['CRITIC'])
        self.target_critic1 = network_maker1(self.state_size, self.action_size, cfg=cfg['RL']['NETWORK']['CRITIC'])
        self.target_critic2 = network_maker1(self.state_size, self.action_size, cfg=cfg['RL']['NETWORK']['CRITIC'])
        self.actor          = network_maker1(self.state_size, self.action_size, cfg=cfg['RL']['NETWORK']['ACTOR'])
        self.target_actor   = network_maker1(self.state_size, self.action_size, cfg=cfg['RL']['NETWORK']['ACTOR'])
        self.log_alpha      = tf.math.log(0.2)
        self.alpha          = tf.math.exp(self.log_alpha)
        self.target_entropy = -tf.convert_to_tensor(np.array(self.action_size,dtype=np.float32),dtype=tf.float32)
        # Optimizer
        self.critic1_optimizer  = Adam(lr=self.critic_lr)
        self.critic2_optimizer  = Adam(lr=self.critic_lr)
        self.actor_optimizer    = Adam(lr=self.actor_lr)
        self.hard_update_target_model()

        # Miscellaneous
        self.update_freq = 1
        self.step_idx = 0
        self.train_idx = 0
        self.train_per_epi = 2
        self.show_media_info = False

        print(self.filename)
        print('States {0}, Actions {1}'.format(self.state_size, self.action_size))

    def get_action(self, state):
        self.step_idx += 1
        state   = tf.convert_to_tensor([state], dtype=tf.float32)
        mu, std = self.actor(state)
        action, log_prob = self.eval_action(mu[0], std[0])
        return action.numpy()

    def remember(self, states, action, reward, next_states, done, goal=None):
        states      = np.array(states,          dtype=np.float32)
        action      = np.array(action,          dtype=np.float32)
        reward      = np.array([reward],        dtype=np.float32)
        done        = np.array([done],          dtype=np.float32)
        next_states = np.array(next_states,     dtype=np.float32)

        if self.er_type == "HER":
            goal        = np.array(goal,        dtype=np.float32)
            transition  = (states, action, reward, next_states, done, goal)
        else:
            transition  = (states, action, reward, next_states, done)
        self.memory.append(transition)
        return

    def eval_action(self, mu, std, epsilon=1e-6):
        action_prob = tfd.Normal(loc=mu, scale=std)
        z = action_prob.sample()
        action = tf.math.tanh(z)
        log_prob = action_prob.log_prob(z) - tf.math.log(1.0 - tf.pow(action,2) + epsilon)
        log_prob = tf.reduce_sum(log_prob, axis=-1, keepdims=True)
        return action, log_prob

    def train_model(self,done=False):
        # Train from Experience Replay
        # Training Condition - Memory Size
        if len(self.memory) < self.train_start:
            return False, 0.0,0.0,0.0
        # if done == False:
        #     return False, 0.0,0.0,0.0
        start_time = time.time()
        for e in range(self.train_per_epi):
            critic_loss_out, actor_loss_out = self._train_model()
        end_time = time.time()
        self.train_consuming_time = end_time - start_time

        return True, critic_loss_out, actor_loss_out, self.train_consuming_time

    def _train_model(self):
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
            print('**** shape of mini_batch',   np.shape(mini_batch),   type(mini_batch))
            print('**** shape of states',       np.shape(states),       type(states))
            print('**** shape of actions',      np.shape(actions),      type(actions))
            print('**** shape of rewards',      np.shape(rewards),      type(rewards))
            print('**** shape of next_states',  np.shape(next_states),  type(next_states))
            print('**** shape of dones',        np.shape(dones),        type(dones))
            if self.er_type == "HER":
                goals = tf.convert_to_tensor(np.array([sample[5] for sample in mini_batch]))
                print('**** shape of goals', np.shape(goals),type(goals))
        # print(' shape : ',np.shape())
        # Update critic
        mu, std = self.actor(next_states,training=True)
        next_actions, next_log_pi = self.eval_action(mu, std)
        target_q1 = self.target_critic1([next_states, next_actions],training=True)
        target_q2 = self.target_critic2([next_states, next_actions],training=True)
        target_q_min = tf.minimum(target_q1, target_q2) # Clipping Double Q
        target_value = rewards + (1.0 - dones) * self.discount_factor * (target_q_min - self.alpha * next_log_pi)

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

        # Update actor
        with tf.GradientTape() as tape:
            mu, std = self.actor(next_states,training=True)
            new_actions, new_log_pi = self.eval_action(mu,std)
            new_q1 = self.critic1([states, new_actions],training=True)
            new_q2 = self.critic2([states, new_actions],training=True)
            new_q_min   = tf.minimum(new_q1, new_q2)
            actor_loss  = tf.reduce_mean(self.alpha * new_log_pi - new_q_min)
        actor_loss_out = actor_loss.numpy()
        actor_params = self.actor.trainable_variables
        actor_grads = tape.gradient(actor_loss, actor_params)
        self.actor_optimizer.apply_gradients(zip(actor_grads, actor_params))

        # Update alpha
        # with tf.GradientTape() as tape:
        #     alpha_loss = - tf.reduce_mean(self.log_alpha * (new_log_pi + self.target_entropy))
        # alpha_params = self.log_alpha
        # alpha_grads = tape.gradient(alpha_loss, alpha_params)
        # self.alpha_optimizer.apply_gradients(zip(alpha_grads, alpha_params))
        # self.alpha = tf.math.exp(self.log_alpha)

        if self.er_type == "PER":
            sample_importance = np.squeeze(td_error.numpy())
            for i in range(self.batch_size):
                self.memory.update(idxs[i], sample_importance[i])

        critic_loss_out = 0.5*(critic1_loss.numpy() + critic2_loss.numpy())
        return critic_loss_out, actor_loss_out

    def update_model(self,done=False):
        if self.train_idx % self.update_freq == 0:
            self.soft_update_target_model()
        return

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

    def get_name(self):
        return 'SAC_1'

################################################################################################
################################################################################################
################################################################################################
################################################################################################
################################################################################################

class SACAgent2:
    def __init__(self, env:object, cfg:dict):
        self.state_size = (env.observation_space[0].shape, env.observation_space[1].shape)
        self.action_size= env.action_space.shape
        # self.board      = board
        self.env_name   = cfg["ENV"]['NAME']
        self.rl_type    = cfg["RL"]['TYPE']
        self.er_type    = cfg["ER"]["ALGORITHM"].upper()
        rl_name = cfg["RL"]["ALGORITHM"]
        for item in cfg['RL']['TYPE']:
            rl_name = rl_name + '_' + item
        self.filename   = self.env_name + '_' + rl_name + '_' + self.er_type
        if cfg["ER"]["ALGORITHM"] == "HER":
            self.filename = self.filename + '_' + cfg["ER"]["STRATEGY"]
        for item in cfg["ADD_NAME"]:
            self.filename = self.filename + '_' + item

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
        self.critic_lr  = 0.002
        self.actor_lr   = 0.001
        self.alpha_lr   = 0.001
        self.tau        = 0.005
        self.alpha      = 0.200 # temperature

        # Networks
        cfg['RL']['NETWORK']['ACTOR']['ACTION_TYPE'] = 'STOCHASTIC'
        self.critic1        = network_maker2(self.state_size, self.action_size, cfg=cfg['RL']['NETWORK']['CRITIC'])
        self.critic2        = network_maker2(self.state_size, self.action_size, cfg=cfg['RL']['NETWORK']['CRITIC'])
        self.target_critic1 = network_maker2(self.state_size, self.action_size, cfg=cfg['RL']['NETWORK']['CRITIC'])
        self.target_critic2 = network_maker2(self.state_size, self.action_size, cfg=cfg['RL']['NETWORK']['CRITIC'])
        self.actor          = network_maker2(self.state_size, self.action_size, cfg=cfg['RL']['NETWORK']['ACTOR'])
        self.target_actor   = network_maker2(self.state_size, self.action_size, cfg=cfg['RL']['NETWORK']['ACTOR'])
        self.log_alpha      = tf.math.log(0.2)
        self.alpha          = tf.math.exp(self.log_alpha)
        self.target_entropy = -tf.convert_to_tensor(np.array(self.action_size,dtype=np.float32),dtype=tf.float32)
        # Optimizer
        self.critic1_optimizer  = Adam(lr=self.critic_lr)
        self.critic2_optimizer  = Adam(lr=self.critic_lr)
        self.actor_optimizer    = Adam(lr=self.actor_lr)
        self.hard_update_target_model()

        # Miscellaneous
        self.update_freq = 1
        self.step_idx = 0
        self.train_idx = 0
        self.train_per_epi = 2
        self.show_media_info = False

        print(self.filename)
        print('States {0}, Actions {1}'.format(self.state_size, self.action_size))

    def get_action(self, state):
        self.step_idx += 1
        video   = tf.convert_to_tensor([state[0]], dtype=tf.float32)
        feature = tf.convert_to_tensor([state[1]], dtype=tf.float32)
        state   = [video, feature]
        mu, std = self.actor(state)
        action, log_prob = self.eval_action(mu[0], std[0])
        return np.clip(action.numpy(), a_min=-1.0, a_max=1.0)

    def remember(self, states, action, reward, next_states, done, goal=None):
        video       = np.array(states[0],       dtype=np.float32)
        feature     = np.array(states[1],       dtype=np.float32)
        action      = np.array(action,          dtype=np.float32)
        reward      = np.array([reward],        dtype=np.float32)
        done        = np.array([done],          dtype=np.float32)
        next_video  = np.array(next_states[0],  dtype=np.float32)
        next_feature= np.array(next_states[1],  dtype=np.float32)
        if self.er_type == "HER":
            goal        = np.array(goal,        dtype=np.float32)
            transition  = (video, feature, action, reward, next_video, next_feature, done, goal)
        else:
            transition  = (video, feature, action, reward, next_video, next_feature, done)
        self.memory.append(transition)
        return

    def eval_action(self, mu, std, epsilon=1e-6):
        action_prob = tfd.Normal(loc=mu, scale=std)
        z = action_prob.sample()
        action = tf.math.tanh(z)
        log_prob = action_prob.log_prob(z) - tf.math.log(1.0 - tf.pow(action,2) + epsilon)
        log_prob = tf.reduce_sum(log_prob, axis=-1, keepdims=True)
        return action, log_prob

    def train_model(self,done=False):
        # Train from Experience Replay
        # Training Condition - Memory Size
        if len(self.memory) < self.train_start:
            return False, 0.0,0.0,0.0
        # if done == False:
        #     return False, 0.0,0.0,0.0
        start_time = time.time()
        for e in range(self.train_per_epi):
            critic_loss_out, actor_loss_out = self._train_model()
        end_time = time.time()
        self.train_consuming_time = end_time - start_time

        return True, critic_loss_out, actor_loss_out, self.train_consuming_time

    def _train_model(self):
        self.train_idx = self.train_idx + 1
        # Sampling from the memory
        if self.er_type == "ER" or self.er_type == "HER":
            mini_batch = self.memory.sample(self.batch_size)
        elif self.er_type == "PER":
            mini_batch, idxs, is_weights = self.memory.sample(self.batch_size)

        videos        = tf.convert_to_tensor(np.array([sample[0] for sample in mini_batch]))
        features      = tf.convert_to_tensor(np.array([sample[1] for sample in mini_batch]))
        actions       = tf.convert_to_tensor(np.array([sample[2] for sample in mini_batch]))
        rewards       = tf.convert_to_tensor(np.array([sample[3] for sample in mini_batch]))
        next_videos   = tf.convert_to_tensor(np.array([sample[4] for sample in mini_batch]))
        next_features = tf.convert_to_tensor(np.array([sample[5] for sample in mini_batch]))
        dones         = tf.convert_to_tensor(np.array([sample[6] for sample in mini_batch]))
        
        if self.show_media_info == False:
            self.show_media_info = True
            print('Start to train, check batch shapes')
            print('**** shape of mini_batch', 	np.shape(mini_batch),	type(mini_batch))
            print('**** shape of videos', 		np.shape(videos),		type(videos))
            print('**** shape of features', 	np.shape(features),		type(features))
            print('**** shape of actions', 		np.shape(actions),		type(actions))
            print('**** shape of rewards', 		np.shape(rewards),		type(rewards))
            print('**** shape of dones', 		np.shape(dones),		type(dones))
            if self.er_type == "HER":
                goals = tf.convert_to_tensor(np.array([sample[7] for sample in mini_batch]))
                print('**** shape of goals', np.shape(goals),type(goals))
        # print(' shape : ',np.shape())
        # Update critic
        mu, std = self.actor([next_videos, next_features],training=True)
        next_actions, next_log_pi = self.eval_action(mu, std)
        target_q1 = self.target_critic1([next_videos, next_features, next_actions],training=True)
        target_q2 = self.target_critic2([next_videos, next_features, next_actions],training=True)
        target_q_min = tf.minimum(target_q1, target_q2) # Clipping Double Q
        target_value = rewards + (1.0 - dones) * self.discount_factor * (target_q_min - self.alpha * next_log_pi)

        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            q1 = self.critic1([videos, features, actions], training=True)
            q2 = self.critic2([videos, features, actions], training=True)
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

        # Update actor
        with tf.GradientTape() as tape:
            mu, std = self.actor([next_videos, next_features],training=True)
            new_actions, new_log_pi = self.eval_action(mu,std)
            new_q1 = self.critic1([videos, features, new_actions],training=True)
            new_q2 = self.critic2([videos, features, new_actions],training=True)
            new_q_min   = tf.minimum(new_q1, new_q2)
            actor_loss  = tf.reduce_mean(self.alpha * new_log_pi - new_q_min)
        actor_loss_out = actor_loss.numpy()
        actor_params = self.actor.trainable_variables
        actor_grads = tape.gradient(actor_loss, actor_params)
        self.actor_optimizer.apply_gradients(zip(actor_grads, actor_params))

        # Update alpha
        # with tf.GradientTape() as tape:
        #     alpha_loss = - tf.reduce_mean(self.log_alpha * (new_log_pi + self.target_entropy))
        # alpha_params = self.log_alpha
        # alpha_grads = tape.gradient(alpha_loss, alpha_params)
        # self.alpha_optimizer.apply_gradients(zip(alpha_grads, alpha_params))
        # self.alpha = tf.math.exp(self.log_alpha)

        if self.er_type == "PER":
            sample_importance = np.squeeze(td_error.numpy())
            for i in range(self.batch_size):
                self.memory.update(idxs[i], sample_importance[i])

        critic_loss_out = 0.5*(critic1_loss.numpy() + critic2_loss.numpy())
        return critic_loss_out, actor_loss_out

    def update_model(self,done=False):
        if self.train_idx % self.update_freq == 0:
            self.soft_update_target_model()
        return

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

    def get_name(self):
        return 'SAC2'