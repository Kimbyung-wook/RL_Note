# Find RL_Note path and append sys path
import os, sys
cwd = os.getcwd()
dir_name = 'RL_note'
pos = cwd.find(dir_name)
root_path = cwd[0:pos] + dir_name
sys.path.append(root_path)
print(root_path)
workspace_path = root_path + "\\pys"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import gym
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Reshape, LeakyReLU, ReLU
from tensorflow.keras.layers import Flatten, Dropout, BatchNormalization
from tensorflow.keras.layers import MaxPool2D, UpSampling2D, Conv2D, Conv2DTranspose
from tensorflow.keras.activations import tanh as Tanh
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
from tensorflow.keras.optimizers import Adam, RMSprop
# from tensorflow.keras.metrics import Accuracy, MeanSquaredError
from tensorflow.keras.callbacks import ModelCheckpoint

from pys.utils.memory import ReplayMemory
from pys.utils.prioritized_memory import ProportionalPrioritizedMemory
from pys.utils.hindsight_memory import HindsightMemory
from pys.model.q_network import QNetwork
from pys.env_config import env_configs

img_size = (128,96)

def get_encoder(input_shape, compressed_shape):
    X_input = Input(shape=input_shape, name='Input')
    X = X_input
    X = Conv2D(filters=32, kernel_size=4, strides=2, padding='SAME', \
                data_format="channels_last", name='Encoder1')(X)
    X = BatchNormalization(name="BN1")(X)
    X = ReLU(name='Relu1')(X)
    X = MaxPool2D(          pool_size=2,padding='SAME', name='MaxPool1')(X)

    X = Conv2D(filters=64, kernel_size=4, strides=2, padding='SAME', \
                data_format="channels_last", name='Encoder2')(X)
    X = BatchNormalization(name="BN2")(X)
    X = ReLU(name='Relu2')(X)
    X = MaxPool2D(          pool_size=2,padding='SAME', name='MaxPool2')(X)

    X = Conv2D(filters=128, kernel_size=2, strides=1, padding='SAME', \
                data_format="channels_last", name='Encoder3')(X)
    X = ReLU(name='Relu3')(X)

    X = Flatten(name='Flattening')(X)
    encoder_output = Dense(compressed_shape[0], activation='linear', \
                name='EncoderOut')(X)
    encoder_model = Model(inputs=X_input, outputs=encoder_output, name='Encoder')

    return encoder_model

def get_decoder(input_shape, compressed_shape):
    X_input = Input(shape=compressed_shape, name='Input')
    X = X_input

    X = Dense(units=6*8*128, activation='linear', \
                name='DeFlattening')(X)
    X = Reshape((6,8,128), name='Reshape')(X)

    X = Conv2DTranspose(filters=128, kernel_size=2, strides=1, padding='SAME', \
                data_format="channels_last", name='Decoder1')(X)
    X = BatchNormalization(name="BN1")(X)
    X = ReLU(name='Relu1')(X)
    X = UpSampling2D(          size=2, interpolation='nearest',\
                data_format="channels_last", name='UpSampling1')(X)

    X = Conv2DTranspose(filters=64, kernel_size=4, strides=2, padding='SAME', \
                data_format="channels_last", name='Decoder2')(X)
    X = BatchNormalization(name="BN2")(X)
    X = ReLU(name='Relu2')(X)
    X = UpSampling2D(          size=2, interpolation='nearest',\
                data_format="channels_last", name='UpSampling2')(X)

    X = Conv2DTranspose(filters=32, kernel_size=4, strides=2, padding='SAME', \
                data_format="channels_last", name='Decoder3')(X)
    X = ReLU(name='Relu3')(X)

    # X = Conv2DTranspose(filters=32, kernel_size=8, strides=4, padding='SAME', \
    #             data_format="channels_last", name='Decoder4')(X)
    # X = BatchNormalization(name="BN4")(X)
    # X = LeakyReLU(name='ReLU4')(X)

    decoder_output = Conv2DTranspose(filters=1, kernel_size=4, strides=1, padding='SAME', \
                data_format="channels_last", activation='tanh', name='Decoder_out')(X)
    decoder_model = Model(inputs=X_input, outputs=decoder_output, name='Decoder')
    
    return decoder_model

def AutoEncoder(input_shape, compressed_shape):
        input_shape = input_shape
        compressed_shape = compressed_shape

        # Hyper-parameter
        learning_rate = 0.001
        batch_size = 32

        # Define auto-encoder
        print('input shape : ',input_shape)
        print('compressed_shape : ',compressed_shape)
        encoder = get_encoder(input_shape=input_shape, compressed_shape=compressed_shape)
        decoder = get_decoder(input_shape=input_shape, compressed_shape=compressed_shape)
        # encoder.compile(optimizer=Adam(learning_rate=learning_rate), loss=MeanSquaredError())
        # decoder.compile(optimizer=Adam(learning_rate=learning_rate), loss=MeanSquaredError())
        # encoder.summary()
        # decoder.summary()

        # Connect encoder with decoder
        encoder_in = Input(shape=input_shape)
        decoder_out= decoder(encoder(encoder_in))
        
        auto_encoder = Model(inputs=encoder_in, outputs=decoder_out)
        auto_encoder.compile(   optimizer=Adam(learning_rate=learning_rate),\
                                loss=MeanSquaredError(),\
                                metrics=[['accuracy'], ['mse']])
        # auto_encoder.compile(optimizer=Adam(learning_rate=learning_rate), loss=BinaryCrossentropy())
        auto_encoder.summary()
        return auto_encoder, encoder, decoder

class DQNAgent:
    def __init__(self, env:object, cfg:dict):
        self.state_size = (img_size[1], img_size[0],1)
        self.action_size= env.action_space.n
        self.env_name   = cfg["ENV"]
        self.rl_type    = "DQN"
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

        # Hyper-parameters for learning
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.tau = 0.005
        
        # Auto Encoder
        self.compressed_shape = (4,)
        self.auto_encoder, self.encoder, self.decoder = AutoEncoder(self.state_size, self.compressed_shape)
        self.train_period_ae = 1000
        self.batch_size_ae = 1000
        self.do_train_ae = False
        self.is_fit = True
        self.hist = None
        self.fit_criterion = 0.98
        # self.image_cols = self.state_size[1]
        # self.image_rows = self.state_size[2]

        # Neural Network Architecture
        self.model        = QNetwork(self.compressed_shape, self.action_size, cfg["RL"]["NETWORK"])
        self.target_model = QNetwork(self.compressed_shape, self.action_size, cfg["RL"]["NETWORK"])
        self.optimizer = tf.keras.optimizers.Adam(lr=self.learning_rate)
        self.hard_update_target_model()
        
        # Miscellaneous
        self.show_media_info = False
        self.steps = 0
        self.update_period = 100

        print(self.filename)
        print('States {0}, Actions {1}'.format(self.state_size, self.action_size))
        
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
            feature = self.encoder.predict(state)
            return np.argmax(self.model(feature))
        
    def train_model(self):
        # Train from Experience Replay
        # Training Condition - Memory Size
        if len(self.memory) < self.train_start:
            return 0.0
        # For Auto-Encoder
        if (self.steps % self.train_period_ae == 0):
            self.do_train_ae = True
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
        feature         = self.encoder.predict(states)
        next_feature    = self.encoder.predict(next_states)
        
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
            tape.watch(model_params)
            # get q value
            q = self.model(feature)
            one_hot_action = tf.one_hot(actions, self.action_size)
            q = tf.reduce_sum(one_hot_action * q, axis=1)
            q = tf.expand_dims(q,axis=1)
            # Target q and maximum target q
            target_q = tf.stop_gradient(self.target_model(next_feature))
            max_q = tf.reduce_max(target_q,axis=1)
            max_q = tf.expand_dims(max_q,axis=1)
            
            targets = rewards + (1 - dones) * self.discount_factor * max_q
            td_error = targets - q
            loss = tf.reduce_mean(tf.square(targets - q))
            
        grads = tape.gradient(loss, model_params)
        self.optimizer.apply_gradients(zip(grads, model_params))

        if self.er_type == "PER":
            sample_importance = td_error.numpy()
            for i in range(self.batch_size):
                self.memory.update(idxs[i], sample_importance[i])

        return loss

    def train_auto_encoder(self):
        # Check model accurac for training
        if self.is_fit:
            # Sampling from the memory
            if self.er_type == "ER" or self.er_type == "HER":
                mini_batch = self.memory.sample(self.batch_size_ae)
            elif self.er_type == "PER":
                mini_batch, _, _ = self.memory.sample(self.batch_size_ae)
            x_train = tf.convert_to_tensor(np.array([sample[0] for sample in mini_batch]))

            evaluated = self.auto_encoder.evaluate(x_train,x_train,verbose=2)
            # print(evaluated)
            loss = evaluated[0]; acc = evaluated[1]
            print('Evaluation loss {:.6f}, accuracy {:.4f} %'.format(loss,acc*100.0))
            if acc < self.fit_criterion:
                print('Re-train Autoencoder')
                self.is_fit = False
            else:
                return
            
        print('***** Train Auto-Encoder *****')
        # Sampling from the memory
        if self.er_type == "ER" or self.er_type == "HER":
            mini_batch = self.memory.sample(self.batch_size_ae)
        elif self.er_type == "PER":
            mini_batch, idxs, is_weights = self.memory.sample(self.batch_size_ae)
        x_train = tf.convert_to_tensor(np.array([sample[0] for sample in mini_batch]))
        checkpoint_path = 'cartpole_auto_encoder.ckpt'
        # Train
        checkpoint = ModelCheckpoint(checkpoint_path, 
                                    save_best_only=True, 
                                    save_weights_only=True, 
                                    monitor='loss', 
                                    verbose=1)
        hist = self.auto_encoder.fit(x_train, x_train, 
                                    # batch_size=self.batch_size_ae, 
                                    epochs=20, 
                                    callbacks=[checkpoint], 
                                    validation_split=0.1,
                                    verbose=1
                                    )
        fig = plt.figure(2)
        loss_ax = plt.subplot()
        acc_ax  = plt.twinx()
        loss_ax.plot(hist.history['loss'],label='loss'); loss_ax.plot(hist.history['val_loss'],label='val_loss')
        loss_ax.set_xlabel('episode'); loss_ax.set_ylabel('loss')
        acc_ax.plot(hist.history['accuracy'],label='accuracy');acc_ax.plot(hist.history['val_accuracy'],label='val_accuracy')
        acc_ax.set_ylabel('Accuracy')
        plt.legend()
        plt.grid(); plt.title('Learning Process of Auto-Encoder')
        plt.savefig('Learning Process of Auto-Encoder.jpg')
        self.auto_encoder.save_weights(checkpoint_path)
        print('Save model weights')
        if hist.history['accuracy'][-1] > self.fit_criterion:    
            print('Accuracy of Auto-Encoder is {:.2f} %, over {:.1f} %'\
                .format(hist.history['accuracy'][-1]*100.0,self.fit_criterion*100.0))
            self.is_fit = True
        return

    def step_update(self, done:bool = False):
        if self.steps % self.update_period == 0:
            self.soft_update_target_model()

        if done == True:
            self.episode_update()
        return

    def episode_update(self):
        if self.do_train_ae == True:
            self.do_train_ae = False
            self.train_auto_encoder()
        return

    def load_model(self,at:str):
        self.model.load_weights( at + self.filename + "_TF")
        self.target_model.load_weights(at + self.filename + "_TF")
        return

    def save_model(self,at:str):
        self.model.save_weights( at + self.filename + "_TF", save_format="tf")
        self.target_model.save_weights(at + self.filename + "_TF", save_format="tf")
        return

def get_image(img_rgb):
    img_rgb_resize = cv2.resize(img_rgb, (img_size[0],img_size[1]), interpolation=cv2.INTER_CUBIC)
    img_k_resize = cv2.cvtColor(img_rgb_resize,cv2.COLOR_RGB2GRAY)
    img_k_resize = img_k_resize / 255.0
    state = img_k_resize
    return state

if __name__ == "__main__":
    cfg = {\
            # "ENV":"Pong-v0",\
            "ENV":"CartPole-v1",\
            "RL":{
                "ALGORITHM":"DQN",\
                "NETWORK":{
                    "LAYER":[128,128],\
                }
            },\
            "ER":
                {
                    "ALGORITHM":"ER",\
                    "REPLAY_N":64,\
                    "STRATEGY":"EPISODE",\
                    # "REWARD_FUNC":reward_function,\
                    # "DONE_FUNC":done_function,\
                },\
            "BATCH_SIZE":32,\
            "TRAIN_START":2000,\
            "MEMORY_SIZE":100000,\
            }
    env_config = env_configs[cfg["ENV"]]
    if cfg["ER"]["ALGORITHM"] == "HER":
        FILENAME = cfg["ENV"] + '_' + cfg["RL"]["ALGORITHM"] + '_' + cfg["ER"]["ALGORITHM"] + '_' + cfg["HER"]["STRATEGY"]
    else:
        FILENAME = cfg["ENV"] + '_' + cfg["RL"]["ALGORITHM"] + '_' + cfg["ER"]["ALGORITHM"]
    EPISODES = env_config["EPISODES"]
    END_SCORE = env_config["END_SCORE"]

    env = gym.make(cfg["ENV"])
    agent = DQNAgent(env, cfg)

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
        state = get_image(env.render(mode='rgb_array'))
        while not done:
            # Interact with env.
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = get_image(env.render(mode='rgb_array'))
            agent.remember(state, action, reward, next_state, done, goal)
            loss = agent.train_model()
            agent.step_update(done)
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
                plt.title(FILENAME)
                plt.subplot(312)
                plt.plot(episodes, epsilons, 'b')
                plt.xlabel('episode'); plt.ylabel('epsilon'); plt.grid()
                plt.subplot(313)
                plt.plot(episodes, losses, 'b')
                plt.xlabel('episode'); plt.ylabel('losses') ;plt.grid()
                plt.savefig(FILENAME + "_TF.jpg", dpi=100)

                # 이동 평균이 0 이상일 때 종료
                if score_avg > END_SCORE:
                    # agent.save_model(workspace_path + "\\result\\save_model\\")
                    end = True
                    break
        if end == True:
            env.close()
            # np.save(workspace_path + "\\result\\data\\" + FILENAME + "_TF_epi",  episodes)
            # np.save(workspace_path + "\\result\\data\\" + FILENAME + "_TF_scores_avg",scores_avg)
            # np.save(workspace_path + "\\result\\data\\" + FILENAME + "_TF_scores_raw",scores_raw)
            # np.save(workspace_path + "\\result\\data\\" + FILENAME + "_TF_losses",losses)
            print("End")
            break