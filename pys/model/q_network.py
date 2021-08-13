import tensorflow as tf
from tensorflow.keras.layers import Dense

class QNetwork(tf.keras.Model):
    def __init__(self, state_size, action_size,cfg):
        super(QNetwork, self).__init__()
        self.structure = cfg["LAYER"]
        self.fcs = []
        for numbers in self.structure:
            fc = Dense(numbers,activation='relu')
            self.fcs.append(fc)
        # self.fc1 = Dense(cfg["RL"]["NETWORK"]["LAYER"][0],activation='relu')
        # self.fc2 = Dense(cfg["RL"]["NETWORK"]["LAYER"][1],activation='relu')
        self.out = Dense(action_size,kernel_initializer=tf.keras.initializers.RandomUniform(-1e-3,1e-3))

    def call(self,x):
        for fc in self.fcs:
            x = fc(x)
        q = self.out(x)
        return q