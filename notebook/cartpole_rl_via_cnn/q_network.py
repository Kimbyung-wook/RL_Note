import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D, Flatten, MaxPool2D, BatchNormalization
from tensorflow.keras.layers import Input
from tensorflow.keras import Model

# def OurModel(input_shape, action_space):
#     X_input = Input(input_shape)
#     # 4 x 160 x 240
#     X = X_input 
#     # print(np.shape(X))
#     X = Conv2D(filters=16, kernel_size=5,   strides=2,  padding="valid",    activation="relu",  data_format="channels_last", input_shape=input_shape)(X)
#     X = BatchNormalization(axis=-1)(X)
#     X = MaxPool2D(           pool_size=3,   strides=2,  padding="valid",                        data_format="channels_last")(X)
#     X = Conv2D(filters=32, kernel_size=4,   strides=2,  padding="valid",    activation="relu",  data_format="channels_last")(X)
#     X = BatchNormalization(axis=-1)(X)
#     X = MaxPool2D(           pool_size=3,   strides=2,  padding="valid",                        data_format="channels_last")(X)
#     X = Conv2D(filters=32, kernel_size=3,   strides=2,  padding="valid",    activation="relu",  data_format="channels_last")(X)
#     X = Flatten()(X)
#     X = Dense(128,                                                          activation="relu", kernel_initializer='he_uniform')(X)
#     X = Dense(64,                                                           activation="relu", kernel_initializer='he_uniform')(X)
#     X = Dense(32,                                                           activation="relu", kernel_initializer='he_uniform')(X)
#     X = Dense(action_space, activation="linear", kernel_initializer='he_uniform')(X)

#     # model = Model(inputs = X_input, outputs = X, name='CartPole PER D3QN CNN model')
#     model = Model(inputs = X_input, outputs = X)
#     model.compile(loss="mean_squared_error", optimizer=RMSprop(lr=0.001, rho=0.95, epsilon=0.01), metrics=["accuracy"])

#     # model.summary()
#     return model

# Multi-Layer Perceptron
class MLPNetwork(tf.keras.Model):
    def __init__(self, state_size, action_size,cfg):
        super(MLPNetwork, self).__init__()
        self.structure = cfg["LAYER"]
        self.fcs = []
        for numbers in self.structure:
            fc = Dense(numbers,activation='relu')
            self.fcs.append(fc)
        self.out = Dense(action_size,kernel_initializer=tf.keras.initializers.RandomUniform(-1e-3,1e-3))

    def call(self,x):
        for fc in self.fcs:
            x = fc(x)
        q = self.out(x)
        return q

class CNNNetwork(tf.keras.Model):
    def __init__(self, state_size, action_size,cfg):
        super(CNNNetwork, self).__init__()
        self.structure = cfg["LAYER"]
        self.fcs = []
        fc = Conv2D(filters=32, kernel_size=8, strides=4, padding='valid', activation='relu', data_format='channels_last'); self.fcs.append(fc)
        fc = Conv2D(filters=64, kernel_size=4, strides=2, padding='valid', activation='relu', data_format='channels_last'); self.fcs.append(fc)
        fc = Conv2D(filters=32, kernel_size=3, strides=1, padding='valid', activation='relu', data_format='channels_last'); self.fcs.append(fc)
        fc = Flatten(); self.fcs.append(fc)
        fc = Dense(units=512, activation='relu',kernel_initializer='he_uniform'); self.fcs.append(fc)
        # fc = Dense(units=64, activation='relu',kernel_initializer='he_uniform'); self.fcs.append(fc)
        self.out = Dense(action_size,kernel_initializer=tf.keras.initializers.RandomUniform(-1e-3,1e-3))

    def call(self,x):
        for fc in self.fcs:
            x = fc(x)
        q = self.out(x)
        return q

class CNN1Network(tf.keras.Model):
    # Pytorch Example about RL q-learning
    # https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    def __init__(self, state_size, action_size,cfg):
        super(CNN1Network, self).__init__()
        self.structure = cfg["LAYER"]
        self.fcs = []
        fc = Conv2D(filters=16, kernel_size=5, strides=2, padding='valid', activation='relu', data_format='channels_last'); self.fcs.append(fc)
        fc = BatchNormalization()
        fc = Conv2D(filters=32, kernel_size=5, strides=2, padding='valid', activation='relu', data_format='channels_last'); self.fcs.append(fc)
        fc = BatchNormalization()
        fc = Conv2D(filters=32, kernel_size=5, strides=2, padding='valid', activation='relu', data_format='channels_last'); self.fcs.append(fc)
        fc = BatchNormalization()
        fc = Flatten(); self.fcs.append(fc)
        fc = Dense(units=512, activation='relu',kernel_initializer='he_uniform'); self.fcs.append(fc)
        # fc = Dense(units=64, activation='relu',kernel_initializer='he_uniform'); self.fcs.append(fc)
        self.out = Dense(action_size,kernel_initializer=tf.keras.initializers.RandomUniform(-1e-3,1e-3))

    def call(self,x):
        for fc in self.fcs:
            x = fc(x)
        q = self.out(x)
        return q

class CNN2Network(tf.keras.Model):
    # My Custom
    def __init__(self, state_size, action_size,cfg):
        super(CNN2Network, self).__init__()
        self.structure = cfg["LAYER"]
        self.fcs = []
        fc = Conv2D(filters=16, kernel_size=4, strides=1, padding='valid', activation='relu', data_format='channels_last'); self.fcs.append(fc)
        fc = MaxPool2D(pool_size=2, strides=1, padding='valid', data_format='channels_last'); self.fcs.append(fc)
        fc = Conv2D(filters=32, kernel_size=4, strides=1, padding='valid', activation='relu', data_format='channels_last'); self.fcs.append(fc)
        fc = MaxPool2D(pool_size=2, strides=1, padding='valid', data_format='channels_last'); self.fcs.append(fc)
        fc = Conv2D(filters=64, kernel_size=4, strides=1, padding='valid', activation='relu', data_format='channels_last'); self.fcs.append(fc)
        fc = MaxPool2D(pool_size=2, strides=1, padding='valid', data_format='channels_last'); self.fcs.append(fc)
        fc = Flatten(); self.fcs.append(fc)
        fc = Dense(units=512, activation='relu',kernel_initializer='he_uniform'); self.fcs.append(fc)
        # fc = Dense(units=64, activation='relu',kernel_initializer='he_uniform'); self.fcs.append(fc)
        self.out = Dense(action_size,kernel_initializer=tf.keras.initializers.RandomUniform(-1e-3,1e-3))

    def call(self,x):
        for fc in self.fcs:
            x = fc(x)
        q = self.out(x)
        return q

# class CNN3Network(tf.keras.Model):
#     # https://pylessons.com/CartPole-PER-CNN/
#     def __init__(self, state_size, action_size,cfg):
#         super(CNN3Network, self).__init__()
#         self.structure = cfg["LAYER"]
#         self.fcs = []
#         fc = Conv2D(filters=64, kernel_size=5, strides=3, padding='valid', activation='relu', data_format='channels_last'); self.fcs.append(fc)
#         fc = Conv2D(filters=64, kernel_size=4, strides=2, padding='valid', activation='relu', data_format='channels_last'); self.fcs.append(fc)
#         fc = Conv2D(filters=64, kernel_size=3, strides=1, padding='valid', activation='relu', data_format='channels_last'); self.fcs.append(fc)
#         fc = Flatten(); self.fcs.append(fc)
#         fc = Dense(units=128, activation='relu',kernel_initializer='he_uniform'); self.fcs.append(fc)
#         fc = Dense(units=64, activation='relu',kernel_initializer='he_uniform'); self.fcs.append(fc)
#         fc = Dense(units=32, activation='relu',kernel_initializer='he_uniform'); self.fcs.append(fc)
#         # fc = Dense(units=64, activation='relu',kernel_initializer='he_uniform'); self.fcs.append(fc)
#         self.out = Dense(action_size,kernel_initializer=tf.keras.initializers.RandomUniform(-1e-3,1e-3))

#     def call(self,x):
#         for fc in self.fcs:
#             x = fc(x)
#         q = self.out(x)
#         return q


def CNN3Network(input_shape, action_space,cfg):
    X_input = Input(input_shape)
    # 4 x 160 x 240
    X = X_input 
    X = Conv2D(filters=64, kernel_size=5, strides=3, padding='valid', activation='relu', data_format='channels_last')(X)
    X = Conv2D(filters=64, kernel_size=4, strides=2, padding='valid', activation='relu', data_format='channels_last')(X)
    X = Conv2D(filters=64, kernel_size=3, strides=1, padding='valid', activation='relu', data_format='channels_last')(X)
    X = Flatten()(X)
    X = Dense(units=128, activation='relu',kernel_initializer='he_uniform')(X)
    X = Dense(units= 64, activation='relu',kernel_initializer='he_uniform')(X)
    X = Dense(units= 32, activation='relu',kernel_initializer='he_uniform')(X)
    X = Dense(units= action_space, activation='relu',kernel_initializer='he_uniform')(X)

    # model = Model(inputs = X_input, outputs = X, name='CartPole PER D3QN CNN model')
    model = Model(inputs = X_input, outputs = X, name='CNN3')
    model.build(input_shape=input_shape)
    # model.compile(loss="mean_squared_error", optimizer=RMSprop(lr=0.001, rho=0.95, epsilon=0.01), metrics=["accuracy"])

    # model.summary()
    return model

class CNN4Network(tf.keras.Model):
    # https://pylessons.com/CartPole-PER-CNN/
    def __init__(self, state_size, action_size,cfg):
        super(CNN4Network, self).__init__()
        self.structure = cfg["LAYER"]
        self.fcs = []
        fc = Conv2D(filters=64, kernel_size=5, strides=3, padding='valid', activation='relu', data_format='channels_last'); self.fcs.append(fc)
        fc = MaxPool2D(pool_size=2, strides=1, padding='valid', data_format='channels_last'); self.fcs.append(fc)
        fc = Conv2D(filters=64, kernel_size=4, strides=2, padding='valid', activation='relu', data_format='channels_last'); self.fcs.append(fc)
        fc = MaxPool2D(pool_size=2, strides=1, padding='valid', data_format='channels_last'); self.fcs.append(fc)
        fc = Conv2D(filters=64, kernel_size=3, strides=1, padding='valid', activation='relu', data_format='channels_last'); self.fcs.append(fc)
        fc = MaxPool2D(pool_size=2, strides=1, padding='valid', data_format='channels_last'); self.fcs.append(fc)
        fc = Flatten(); self.fcs.append(fc)
        fc = Dense(units=128, activation='relu',kernel_initializer='he_uniform'); self.fcs.append(fc)
        fc = Dense(units=64, activation='relu',kernel_initializer='he_uniform'); self.fcs.append(fc)
        fc = Dense(units=32, activation='relu',kernel_initializer='he_uniform'); self.fcs.append(fc)
        # fc = Dense(units=64, activation='relu',kernel_initializer='he_uniform'); self.fcs.append(fc)
        self.out = Dense(action_size,kernel_initializer=tf.keras.initializers.RandomUniform(-1e-3,1e-3))

    def call(self,x):
        for fc in self.fcs:
            x = fc(x)
        q = self.out(x)
        return q