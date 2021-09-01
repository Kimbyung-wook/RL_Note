import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense
# Refer from
# https://github.com/gouxiangchen/dueling-DQN-pytorch/blob/master/dueling_dqn_tf2.py
class QNetwork(tf.keras.Model):
    def __init__(self, state_size, action_size,cfg):
        super(QNetwork, self).__init__()
        self.structure = cfg['NETWORK']["LAYER"]
        self.q_type = cfg['TYPE']
        self.fcs = []
        for numbers in self.structure:
            fc = Dense(numbers,activation='relu')
            self.fcs.append(fc)
        self.adv = Dense(action_size,kernel_initializer=tf.keras.initializers.RandomUniform(-1e-3,1e-3))
        if 'DUELING' in self.q_type:
            self.val = Dense(1,kernel_initializer=tf.keras.initializers.RandomUniform(-1e-3,1e-3))
        return

    def call(self,x):
        for fc in self.fcs:
            x = fc(x)
        A = self.adv(x)
        if 'DUELING' in self.q_type:
            V = self.val(x)
            Q = V + A - tf.reduce_mean(A, axis=1, keepdims=True)
        else:
            Q = A
        return Q

def get_q_network(state_space, action_space, cfg):
    structure = cfg['NETWORK']["LAYER"]
    X_input = Input(shape=state_space)
    X = X_input
    for idx in range(len(structure)):
        X = Dense(units=structure[idx], activation='relu',name='Layer'+str(idx))(X)
    A = X
    A = Dense(units=action_space, activation='linear', name='Adv')(A)
    if 'DUELING' in cfg['TYPE']:
        V = X 
        V = Dense(units=1, activation='linear', name='Val')(V)
        Q = V + A - tf.reduce_mean(A, axis=1, keepdims=True)
    else:
        Q = A

    model = Model(inputs = X_input, outputs = Q, name='q_network')
    model.build(input_shape = state_space)

    return model