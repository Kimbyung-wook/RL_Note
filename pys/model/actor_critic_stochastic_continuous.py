import tensorflow as tf
from tensorflow.keras.layers import Dense, concatenate, Lambda

class Actor(tf.keras.Model):
    def __init__(self, state_size, action_size, log_std_min, log_std_max):
        super(Actor, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.fc1 = Dense(64, activation='relu')
        self.fc2 = Dense(64, activation='relu')
        # self.fc3 = Dense(16, activation='relu')
        self.mu  = Dense(action_size)
        self.log_std= Dense(action_size)

    def call(self, x):
        x       = self.fc1(x)
        x       = self.fc2(x)
        # x       = self.fc3(x)
        mu = self.mu(x)
        log_std = self.log_std(x)
        log_std = tf.clip_by_value(log_std, self.log_std_min, self.log_std_max)
        std = tf.math.exp(log_std)
        return mu, std

class Critic(tf.keras.Model):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.s1 = Dense(16, activation='relu')
        self.s2 = Dense(32, activation='relu')
        self.a1 = Dense(32, activation='relu')
        self.a2 = Dense(32, activation='relu')
        self.fc1= Dense(64, activation='relu')
        self.fc2= Dense(64, activation='relu')
        self.out= Dense(1,  activation='linear')

    def call(self,state_action):
        state  = state_action[0]
        action = state_action[1]
        s = self.s1(state)
        s = self.s2(s)
        a = self.a1(action)
        a = self.a2(a)
        c = concatenate([s,a],axis=-1)
        x = self.fc1(c)
        x = self.fc2(x)
        q = self.out(x)
        return q