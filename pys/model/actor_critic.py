import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Concatenate, concatenate
from tensorflow.keras.layers import Conv2D, Flatten, MaxPool2D, BatchNormalization
from tensorflow.keras.utils  import plot_model

def get_actor_mlp(state_space, action_space, cfg):
  structure = cfg['NETWORK']["ACTOR"]
  IN = Input(shape=state_space , name='Input_State')

  X = IN
  for idx in range(len(structure)):
      X = Dense(units=structure[idx], activation='relu',name='Layer'+str(idx))(X)

  MU  = Dense(units=action_space, activation='linear', name='MU')(X)
  STD = Dense(units=action_space, activation='linear', name='STD')(X)
  OUT = [MU, STD]

  model = Model(inputs = IN, outputs = OUT, name='ActorMLP')
  model.build(input_shape = state_space)
  dot_img_file = 'get_actor_mlp.png'
  plot_model(model, to_file=dot_img_file,dpi=100,
            show_shapes=True, show_dtype=True, show_layer_names=True,\
  )
  model.summary()

  return model

def get_critic_mlp(state_space, action_space, cfg):
  layer_s = cfg['NETWORK']["CRITIC"]['STATE']
  layer_a = cfg['NETWORK']["CRITIC"]['ACTION']
  layer_m = cfg['NETWORK']["CRITIC"]['MERGE']

  S = Input(shape=state_space , name='Input_State')
  A = Input(shape=action_space, name='Input_Action')
  IN  = [S, A]

  # State Input
  for idx in range(len(layer_s)):
      S = Dense(units=layer_s[idx], activation='relu',name='State'+str(idx))(S)

  # Action Input
  for idx in range(len(layer_a)):
      A = Dense(units=layer_a[idx], activation='relu',name='Action'+str(idx))(A)
      
  # Signal Merger
  M   = concatenate([S, A],axis=1)
  for idx in range(len(layer_m)):
      M = Dense(units=layer_m[idx], activation='relu',name='Merge'+str(idx))(M)
  Q = Dense(units=1, activation='linear', name='Q')(M)
  OUT = Q

  model = Model(inputs = IN, outputs = OUT, name='CriticMLP')
  model.build(input_shape = state_space)
  dot_img_file = 'get_critic_mlp.png'
  plot_model(model, to_file=dot_img_file,dpi=100,
            show_shapes=True, show_dtype=True, show_layer_names=True,\
  )
  model.summary()

  return model

def get_actor_cnn(state_space, action_space, cfg):
  IN  = Input(shape=state_space, name='Input_Image')

  # Image Input
  S   = IN
  S   = Conv2D(     filters=32, kernel_size=4,  strides=2, padding='valid', data_format='channels_last', name='Conv1', activation='relu')(S)
  S   = MaxPool2D(              pool_size=4,    strides=2, padding='valid', data_format='channels_last', name='MaxPool1')(S)
  S   = Conv2D(     filters=64, kernel_size=2,  strides=1, padding='valid', data_format='channels_last', name='Conv2', activation='relu')(S)
  S   = Flatten()(S)

  M   = S
  M   = Dense(units=64, activation='relu',name='Layer1')(M)
  M   = Dense(units=64, activation='relu',name='Layer2')(M)
  MU  = Dense(units=action_space, activation='linear', name='MU' )(M)
  STD = Dense(units=action_space, activation='linear', name='STD')(M)
  OUT = tf.concat([MU, STD],1)

  model = Model(inputs = IN, outputs = OUT, name='ActorCNN')
  model.build(input_shape = state_space)
  dot_img_file = 'get_actor_cnn.png'
  plot_model(model, to_file=dot_img_file,dpi=100,
            show_shapes=True, show_dtype=True, show_layer_names=True,\
  )
  model.summary()

  return model

def get_critic_cnn(state_space, action_space, cfg):
  S = Input(shape=state_space , name='Input_Image')
  A = Input(shape=action_space, name='Input_Action')
  IN  = [S, A]

  # Image Input
  S   = Conv2D(     filters=32, kernel_size=4,  strides=2, padding='valid', data_format='channels_last', name='Conv1', activation='relu')(S)
  S   = MaxPool2D(              pool_size=4,    strides=2, padding='valid', data_format='channels_last', name='MaxPool1')(S)
  S   = Conv2D(     filters=64, kernel_size=2,  strides=1, padding='valid', data_format='channels_last', name='Conv2', activation='relu')(S)
  S   = Flatten()(S)

  # Action Input
  A   = Dense(units=64, activation='relu',name='Action1')(A)
  A   = Dense(units=64, activation='relu',name='Action2')(A)

  # Signal Merger
  M   = concatenate([S, A],axis=1)
  M   = Dense(units=64, activation='relu',name='Merge1')(M)
  M   = Dense(units=64, activation='relu',name='Merge2')(M)
  Q   = Dense(units=1, activation='linear', name='Q')(M)
  OUT = Q

  model = Model(inputs = IN, outputs = OUT, name='CriticCNN')
  model.build(input_shape = state_space)
  dot_img_file = 'get_critic_cnn.png'
  plot_model(model, to_file=dot_img_file,dpi=100,
            show_shapes=True, show_dtype=True, show_layer_names=True,\
  )
  model.summary()

  return model

def get_actor_cnn_mlp(state_space, action_space, cfg):
  '''
  model = get_actor_cnn_mlp(((64,32,4),4),4,cfg)
  '''
  S1     = Input(shape=state_space[0], name='Input_Image')
  S2     = Input(shape=state_space[1], name='Input_Value')
  IN    = [S1, S2]

  # Image Input
  S1    = S1
  S1    = Conv2D(     filters=32, kernel_size=4,  strides=2, padding='valid', data_format='channels_last', name='Conv1', activation='relu')(S1)
  S1    = MaxPool2D(              pool_size=4,    strides=2, padding='valid', data_format='channels_last', name='MaxPool1')(S1)
  S1    = Conv2D(     filters=64, kernel_size=2,  strides=1, padding='valid', data_format='channels_last', name='Conv2', activation='relu')(S1)
  S1    = Flatten()(S1)

  # Value Input
  S2     = Dense(units=64, activation='relu',name='Feature1')(S2)
  S2     = Dense(units=64, activation='relu',name='Feature2')(S2)

  # Signal Merger
  M     = concatenate([S1, S2],axis=1)
  M     = Dense(units=64, activation='relu',name='Merge1')(M)
  M     = Dense(units=64, activation='relu',name='Merge2')(M)
  MU    = Dense(units=action_space, activation='linear', name='MU' )(M)
  STD   = Dense(units=action_space, activation='linear', name='STD')(M)
  # OUT   = concatenate([MU, STD],1)
  # OUT   = tf.concat([MU, STD],1)
  OUT   = [MU, STD]

  model = Model(inputs = IN, outputs = OUT, name='ActorCNN+MLP')
  model.build(input_shape = state_space)
  dot_img_file = 'get_actor_cnn_mlp.png'
  plot_model(model, to_file=dot_img_file,dpi=100,
            show_shapes=True, show_dtype=True, show_layer_names=True,\
  )
  model.summary()

  return model

def get_critic_cnn_mlp(state_space, action_space, cfg):
  '''
  model = get_critic_cnn_mlp(((64,32,4),4),4,cfg)
  '''
  S1     = Input(shape=state_space[0], name='Input_Image')
  S2     = Input(shape=state_space[1], name='Input_Value')
  A      = Input(shape=action_space,   name='Input_Action')
  IN    = [S1, S2, A]

  # Image Input
  S1    = S1
  S1    = Conv2D(     filters=32, kernel_size=4,  strides=2, padding='valid', data_format='channels_last', name='Conv1', activation='relu')(S1)
  S1    = MaxPool2D(              pool_size=4,    strides=2, padding='valid', data_format='channels_last', name='MaxPool1')(S1)
  S1    = Conv2D(     filters=64, kernel_size=2,  strides=1, padding='valid', data_format='channels_last', name='Conv2', activation='relu')(S1)
  S1    = Flatten()(S1)

  # Value Input
  S2     = Dense(units=64, activation='relu',name='Feature1')(S2)
  S2     = Dense(units=64, activation='relu',name='Feature2')(S2)

  # Action Input
  A     = Dense(units=64, activation='relu',name='Action1')(A)
  A     = Dense(units=64, activation='relu',name='Action2')(A)

  # Signal Merger
  M     = concatenate([S1, S2, A],axis=1)
  M     = Dense(units=64, activation='relu',name='Merge1')(M)
  M     = Dense(units=64, activation='relu',name='Merge2')(M)
  Q     = Dense(units=action_space, activation='linear', name='Q' )(M)
  OUT   = Q

  model = Model(inputs = IN, outputs = OUT, name='CriticCNN+MLP')
  model.build(input_shape = state_space)
  dot_img_file = 'get_critic_cnn_mlp.png'
  plot_model(model, to_file=dot_img_file,dpi=100,
            show_shapes=True, show_dtype=True, show_layer_names=True,\
  )
  model.summary()

  return model

if __name__ == '__main__':
  
  cfg = {
    'NETWORK':{
      'ACTOR':(32,32),
      'CRITIC':{
        'STATE':(16,16),
        'ACTION':(16,16),
        'MERGE':(16,16),
      }
    }
  }
  model = get_actor_mlp( 6,2,cfg)
  model = get_critic_mlp(6,2,cfg)

  model = get_actor_cnn( (64,32,4),4,cfg)
  model = get_critic_cnn((64,32,4),4,cfg)

  model = get_actor_cnn_mlp(((64,32,4),4),4,cfg)
  model = get_critic_cnn_mlp(((64,32,4),4),4,cfg)