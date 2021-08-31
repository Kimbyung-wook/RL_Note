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
        adv = self.adv(x)
        if 'DUELING' in self.q_type:
            val = self.val(x)
            q = val + adv - tf.reduce_mean(adv, axis=1, keepdims=True)
        else:
            q = adv
        return q

def get_q_network(state_space, action_space, cfg):
    X_input = Input(shape=state_space)
    X = X_input
    for item in cfg['NETWORK']["LAYER"]:
        X = Dense(units=item, activation='relu')(X)
    Adv = Dense(units=action_space, activation='relu')(X)
    if 'DUELING' in cfg['TYPE']:
        Val = Dense(units=1, activation='relu')(X)
        Q = Val + Adv - tf.reduce_mean(Adv, axis=1, keepdims=True)
    else:
        Q = Adv

    model = Model(inputs = X_input, outputs = Q, name='q_network')
    model.build(input_shape = state_space)

    return model