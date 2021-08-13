import tensorflow as tf
from tensorflow.keras.layers import Dense, concatenate, Lambda
import tensorflow as tf
from tensorflow.keras.layers import Dense, concatenate, Lambda

class Actor(tf.keras.Model):
    def __init__(self, state_size, action_size, log_std_min, log_std_max,cfg):
        super(Actor, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.structure = cfg["ACTOR"]
        self.layer = []
        for numbers in self.structure:
            layer = Dense(numbers,activation='relu')
            self.layer.append(layer)
        self.out= Dense(1,  activation='linear')
        # self.fc1 = Dense(64, activation='relu')
        # self.fc2 = Dense(64, activation='relu')
        # self.fc3 = Dense(16, activation='relu')

    def call(self, x):
        for layer in self.layer:
            x = layer(x)
        # x       = self.fc1(x)
        # x       = self.fc2(x)
        # x       = self.fc3(x)
        action  = self.out(x)
        a = Lambda(lambda x: x*self.action_max)(action)
        return a

class Critic(tf.keras.Model):
    def __init__(self, state_size, action_size,cfg):
        super(Critic, self).__init__()
        self.structure = cfg["CRITIC"]
        self.state_layer = []
        self.action_layer = []
        self.concat_layer = []
        for numbers in self.structure["STATE"]:
            layer = Dense(numbers,activation='relu')
            self.state_layer.append(layer)
        for numbers in self.structure["ACTION"]:
            layer = Dense(numbers,activation='relu')
            self.action_layer.append(layer)
        for numbers in self.structure["CONCAT"]:
            layer = Dense(numbers,activation='relu')
            self.concat_layer.append(layer)
        self.out= Dense(1,  activation='linear')
        # self.s1 = Dense(16, activation='relu')
        # self.s2 = Dense(32, activation='relu')
        # self.a1 = Dense(32, activation='relu')
        # self.a2 = Dense(32, activation='relu')
        # self.fc1= Dense(64, activation='relu')
        # self.fc2= Dense(64, activation='relu')

    def call(self,state_action):
        s = state_action[0]
        a = state_action[1]
        for state_layer in self.state_layer:
            s = state_layer(s)
        for action_layer in self.action_layer:
            a = action_layer(a)
        x = concatenate([s,a],axis=-1)
        for concat_layer in self.concat_layer:
            x = concat_layer(x)
        q = self.out(x)
        
        # s = self.s1(s)
        # s = self.s2(s)
        # a = self.a1(a)
        # a = self.a2(a)
        # x = concatenate([s,a],axis=-1)
        # x = self.fc1(x)
        # x = self.fc2(x)
        # q = self.out(x)
        return q