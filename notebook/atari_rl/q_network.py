import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input, ReLU, Conv2D, Flatten
from tensorflow.keras.optimizers import Adam

def get_q_network(observation_space, action_space):
  # Input shape is expected as (84,84,4)
  X_input = Input(shape=observation_space)
  # Convolution Layers
  X = X_input
  X = Conv2D(filters=32, kernel_size=8, strides=4, padding='valid', activation="relu", data_format='channels_last')(X)
  X = Conv2D(filters=64, kernel_size=4, strides=2, padding='valid', activation="relu", data_format='channels_last')(X)
  X = Conv2D(filters=64, kernel_size=3, strides=1, padding='valid', activation="relu", data_format='channels_last')(X)
  X = Flatten()(X)
  X = Dense(units=512,          activation='relu',   kernel_initializer='he_uniform')(X)
  X = Dense(units=action_space, activation='linear', kernel_initializer='he_uniform')(X)
  model = Model(inputs=X_input, outputs=X)
  model.build(input_shape=observation_space)

  return model