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
        self.structure = cfg['NETWORK']["LAYER"]
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
        self.structure = cfg['NETWORK']["LAYER"]
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
        self.structure = cfg['NETWORK']["LAYER"]
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
        self.structure = cfg['NETWORK']["LAYER"]
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

# 6000 epi에서 최대 110점
def get_CNN3Network(input_shape, action_space,cfg):
    X_input = Input(input_shape)
    # 120 x 40 x N
    X = X_input 
    X = Conv2D(filters=64, kernel_size=5, strides=3, padding='valid', activation='relu', data_format='channels_last')(X)
    X = Conv2D(filters=64, kernel_size=4, strides=2, padding='valid', activation='relu', data_format='channels_last')(X)
    X = Conv2D(filters=64, kernel_size=3, strides=1, padding='valid', activation='relu', data_format='channels_last')(X)
    X = Flatten()(X)
    X = Dense(units=256, activation='relu',kernel_initializer='he_uniform')(X)
    # X = Dense(units= 64, activation='relu',kernel_initializer='he_uniform')(X)
    # X = Dense(units= 32, activation='relu',kernel_initializer='he_uniform')(X)
    X = Dense(units= action_space, activation='relu',kernel_initializer='he_uniform')(X)

    # model = Model(inputs = X_input, outputs = X, name='CartPole PER D3QN CNN model')
    model = Model(inputs = X_input, outputs = X, name='CNN3')
    model.build(input_shape=input_shape)
    # model.compile(loss="mean_squared_error", optimizer=RMSprop(lr=0.001, rho=0.95, epsilon=0.01), metrics=["accuracy"])

    # model.summary()
    return model

def get_CNN4Network(input_shape, action_space,cfg):
    X_input = Input(input_shape)
    # 120 x 40 x N
    X = X_input 
    X = Conv2D(     filters=64, kernel_size=(6,2),  strides=2, padding='valid', data_format='channels_last', name='Conv1', activation='relu')(X)
    X = MaxPool2D(              pool_size=(6,2),    strides=2, padding='valid', data_format='channels_last', name='MaxPool1')(X)
    X = Conv2D(     filters=64, kernel_size=(6,2),  strides=2, padding='valid', data_format='channels_last', name='Conv2', activation='relu')(X)
    X = MaxPool2D(              pool_size=2,        strides=1, padding='valid', data_format='channels_last', name='MaxPool2')(X)
    X = Conv2D(     filters=64, kernel_size=2,      strides=1, padding='valid', data_format='channels_last', name='Conv3', activation='relu')(X)
    X = Flatten()(X)
    X = Dense(units=256, activation='relu',kernel_initializer='he_uniform')(X)
    A = X
    A = Dense(units= 64,            activation='relu',  kernel_initializer='he_uniform', name='Adv1')(A)
    A = Dense(units = action_space, activation='linear',kernel_initializer='he_uniform', name='Adv2')(A)
    if "DUELING" in cfg['TYPE']:
        print('Define DUELING')
        V = X
        V = Dense(units= 64, activation='relu',  kernel_initializer='he_uniform', name='Val1')(X)
        V = Dense(units = 1, activation='linear',kernel_initializer='he_uniform', name='Val2')(V)
        Q = V + A - tf.reduce_mean(A, axis=1, keepdims=True)
    else:
        Q = A
    model = Model(inputs = X_input, outputs = Q, name='CNN4')
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
        fc = Conv2D(    filters=64, kernel_size=5,  strides=3, padding='valid', data_format='channels_last', activation='relu'); self.fcs.append(fc)
        fc = MaxPool2D(             pool_size=2,    strides=1, padding='valid', data_format='channels_last'); self.fcs.append(fc)
        fc = Conv2D(    filters=64, kernel_size=4,  strides=2, padding='valid', data_format='channels_last', activation='relu'); self.fcs.append(fc)
        fc = MaxPool2D(             pool_size=2,    strides=1, padding='valid', data_format='channels_last'); self.fcs.append(fc)
        fc = Conv2D(    filters=64, kernel_size=3,  strides=1, padding='valid', data_format='channels_last', activation='relu'); self.fcs.append(fc)
        fc = MaxPool2D(             pool_size=2,    strides=1, padding='valid', data_format='channels_last'); self.fcs.append(fc)
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


def get_actor_cnn_mlp(state_space, action_space, cfg):
  '''
  if __name__ == '__main__':
    cfg = {}
    agent = get_actor_cnn_mlp(((64,32,4),4),4,cfg)
    agent.summary()
  '''
  # Image Input
  IN1   = Input(shape=state_space[0], name='Input_Image')
  I     = IN1
  I     = Conv2D(     filters=32, kernel_size=4,  strides=2, padding='valid', data_format='channels_last', name='Conv1', activation='relu')(I)
  I     = MaxPool2D(              pool_size=4,    strides=2, padding='valid', data_format='channels_last', name='MaxPool1')(I)
  I     = Conv2D(     filters=64, kernel_size=2,  strides=1, padding='valid', data_format='channels_last', name='Conv2', activation='relu')(I)
  I     = Flatten()(I)

  # Value Input
  IN2   = Input(shape=state_space[1], name='Input_Value')
  V     = IN2
  V     = Dense(units=64, activation='relu',name='Dense1')(V)

  # Signal Merger
  M     = tf.concat([I, V],axis=1)
  M     = Dense(units=64, activation='relu',name='Merge1')(M)
  M     = Dense(units=64, activation='relu',name='Merge2')(M)
  MU    = Dense(units=action_space, activation='linear', name='MU' )(M)
  STD   = Dense(units=action_space, activation='linear', name='STD')(M)
  OUT   = tf.concat([MU, STD],1)

  IN    = [IN1, IN2]
  model = Model(inputs = IN, outputs = OUT, name='ActorCNN+MLP')
  model.build(input_shape = state_space)

  return model