import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Concatenate

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
        self.mu  = Dense(action_size)
        self.log_std= Dense(action_size)

    def call(self, x):
        for layer in self.layer:
            x = layer(x)
        mu = self.mu(x)
        log_std = self.log_std(x)
        log_std = tf.clip_by_value(log_std, self.log_std_min, self.log_std_max)
        std = tf.math.exp(log_std)
        return mu, std

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

    def call(self,state_action):
        s = state_action[0]
        a = state_action[1]
        for state_layer in self.state_layer:
            s = state_layer(s)
        for action_layer in self.action_layer:
            a = action_layer(a)
        x = Concatenate([s,a],axis=-1)
        for concat_layer in self.concat_layer:
            x = concat_layer(x)
        q = self.out(x)
        
        return q

def get_actor_mlp(state_space, action_space, cfg):
    structure = cfg['NETWORK']["LAYER"]
    X_input = Input(shape=state_space)
    X = X_input
    for idx in range(len(structure)):
        X = Dense(units=structure[idx], activation='relu',name='Layer'+str(idx))(X)
    MU = Dense(units=action_space, activation='linear', name='mu')(X)
    STD = Dense(units=action_space, activation='linear', name='mu')(X)
    OUT = tf.concat([MU, STD])

    model = Model(inputs = X_input, outputs = OUT, name='q_network')
    model.build(input_shape = state_space)

    return model