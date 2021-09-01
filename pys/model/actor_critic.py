import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Concatenate, concatenate
from tensorflow.keras.layers import Conv2D, Flatten, MaxPool2D, BatchNormalization
from tensorflow.keras.utils  import plot_model

def get_actor_mlp(state_space, action_space, cfg):
  '''
  cfg = {
    'NETWORK':{
      'ACTOR':(64,64)
    }
  }
  agent = get_actor_mlp(6,4,cfg)
  '''
  structure = cfg['NETWORK']["ACTOR"]
  IN = Input(shape=state_space)
  # Input
  X = IN
  for idx in range(len(structure)):
      X = Dense(units=structure[idx], activation='relu',name='Layer'+str(idx))(X)
  MU = Dense(units=action_space, activation='linear', name='MU')(X)
  STD = Dense(units=action_space, activation='linear', name='STD')(X)
  # OUT = tf.concat([MU, STD],1)
  OUT = [MU, STD]

  model = Model(inputs = IN, outputs = OUT, name='ActorMLP')
  model.build(input_shape = state_space)
  dot_img_file = 'model_1.png'
  plot_model(model, to_file=dot_img_file,dpi=100,
    show_shapes=True, show_dtype=True, show_layer_names=True
  )
  model.summary()

  return model

def get_actor_cnn(state_space, action_space, cfg):
  '''
  cfg = {
    'NETWORK':{
      'ACTOR':(64,64)
    }
  }
  '''
  IN  = Input(shape=state_space)
  # Input
  X   = IN
  X   = Conv2D(     filters=32, kernel_size=4,  strides=2, padding='valid', data_format='channels_last', name='Conv1', activation='relu')(X)
  X   = MaxPool2D(              pool_size=4,    strides=2, padding='valid', data_format='channels_last', name='MaxPool1')(X)
  X   = Conv2D(     filters=64, kernel_size=2,  strides=1, padding='valid', data_format='channels_last', name='Conv2', activation='relu')(X)
  X   = Flatten()(X)
  X   = Dense(units=64, activation='relu',name='Layer1')(X)
  X   = Dense(units=64, activation='relu',name='Layer2')(X)
  MU  = Dense(units=action_space, activation='linear', name='MU')(X)
  STD = Dense(units=action_space, activation='linear', name='STD')(X)
  OUT = [MU, STD]

  model = Model(inputs = IN, outputs = OUT, name='ActorCNN')
  model.build(input_shape = state_space)
  dot_img_file = 'model_1.png'
  plot_model(model, to_file=dot_img_file,dpi=100,
    show_shapes=True, show_dtype=True, show_layer_names=True
  )
  model.summary()

  return model

def get_actor_cnn_mlp(state_space, action_space, cfg):
  '''
  model = get_actor_cnn_mlp(((64,32,4),4),4,cfg)
  '''
  IN1   = Input(shape=state_space[0], name='Input_Image')
  IN2   = Input(shape=state_space[1], name='Input_Value')
  IN    = [IN1, IN2]

  # Image Input
  I     = IN1
  I     = Conv2D(     filters=32, kernel_size=4,  strides=2, padding='valid', data_format='channels_last', name='Conv1', activation='relu')(I)
  I     = MaxPool2D(              pool_size=4,    strides=2, padding='valid', data_format='channels_last', name='MaxPool1')(I)
  I     = Conv2D(     filters=64, kernel_size=2,  strides=1, padding='valid', data_format='channels_last', name='Conv2', activation='relu')(I)
  I     = Flatten()(I)

  # Value Input
  V     = IN2
  V     = Dense(units=64, activation='relu',name='Dense1')(V)

  # Signal Merger
  M     = concatenate([I, V],axis=1)
  M     = Dense(units=64, activation='relu',name='Merge1')(M)
  M     = Dense(units=64, activation='relu',name='Merge2')(M)
  MU    = Dense(units=action_space, activation='linear', name='MU' )(M)
  STD   = Dense(units=action_space, activation='linear', name='STD')(M)
  # OUT   = concatenate([MU, STD],1)
  # OUT   = tf.concat([MU, STD],1)
  OUT   = [MU, STD]

  model = Model(inputs = IN, outputs = OUT, name='ActorCNN+MLP')
  model.build(input_shape = state_space)
  dot_img_file = 'model_1.png'
  plot_model(model, to_file=dot_img_file,dpi=100,
    show_shapes=True, show_dtype=True, show_layer_names=True
  )
  model.summary()

  return model


import tensorboard
import time
from datetime import datetime
if __name__ == '__main__':
  
  cfg = {
    'NETWORK':{
      'ACTOR':{
      }
    }
  }
  model = get_actor_cnn_mlp(((64,32,4),4),4,cfg)