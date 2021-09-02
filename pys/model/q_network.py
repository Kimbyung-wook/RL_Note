import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Concatenate, concatenate
from tensorflow.keras.layers import Conv2D, Flatten, MaxPool2D, BatchNormalization, AveragePooling2D, SpatialDropout2D
from tensorflow.keras.utils  import plot_model

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
  '''
  cfg={
    'NETWORK':{
      'MLP':(64,64)
    },
    'TYPE':'DUELING',
  }
  model = get_q_network((8,),(4,),cfg)
  model.summary()
  '''
  structure = cfg['NETWORK']["MLP"]
  IN = Input(shape=state_space)
  X = IN
  for idx in range(len(structure)):
    X = Dense(units=structure[idx], activation='relu',name='Layer'+str(idx))(X)
  A = X
  A = Dense(units=action_space[0], activation='linear', name='Adv')(A)
  if 'DUELING' in cfg['TYPE']:
    V = X 
    V = Dense(units=1, activation='linear', name='Val')(V)
    Q = tf.math.add(V, A, name='Add')
    Q = tf.math.subtract(Q,tf.reduce_mean(A, axis=1, keepdims=True), name='Subtract')
  else:
    Q = A

  model = Model(inputs = IN, outputs = Q, name='q_network')
  model.build(input_shape = state_space)
  dot_img_file = 'visualize_model\get_critic_cnn.png'
  plot_model(model, to_file=dot_img_file,dpi=100,
        show_shapes=True, show_dtype=True, show_layer_names=True,\
  )

  return model

def get_dqn_network(state_space, action_space, cfg):
  IN  = Input(shape=state_space, name='Input_Image')

  # Image Input
  S   = IN
  S   = Conv2D(     filters=32, kernel_size=8,  strides=4, padding='valid', data_format='channels_last', name='Conv1', activation='relu')(S)
  S   = Conv2D(     filters=64, kernel_size=4,  strides=2, padding='valid', data_format='channels_last', name='Conv2', activation='relu')(S)
  S   = Conv2D(     filters=64, kernel_size=3,  strides=1, padding='valid', data_format='channels_last', name='Conv3', activation='relu')(S)
  S   = Flatten()(S)
  X   = S
  X = Dense(units=128, activation='relu',name='Layer1')(X)
  X = Dense(units=128, activation='relu',name='Layer2')(X)
  A = Dense(units=action_space[0], activation='linear', name='Adv')(X)
  if 'DUELING' in cfg['TYPE']:
    V = Dense(units=1, activation='linear', name='Val')(X)
    Q = tf.math.add(V, A, name='Add')
    Q = tf.math.subtract(Q,tf.reduce_mean(A, axis=1, keepdims=True), name='Subtract')
  else:
    Q = A

  model = Model(inputs = IN, outputs = Q, name='q_network')
  model.build(input_shape = state_space)
  # dot_img_file = 'visualize_model\get_dqn_network.png'
  # plot_model(model, to_file=dot_img_file,dpi=100,
  #       show_shapes=True, show_dtype=True, show_layer_names=True,\
  # )
  return model

def get_mlp_network(state_space, action_space, cfg):
  '''
  cfg={
    'NETWORK':{
      'MLP':(
        (64,'relu'),
        (64,'relu'),
        (64,'relu'),
      )
    },
    'TYPE':'',
  }
  model = get_mlp_network((8,),4,cfg)
  model.summary()
  '''
  mlp_layer = cfg['NETWORK']["MLP"]
  IN  = Input(shape=state_space, name='Input_Image')

  # MLP Layers
  X = IN
  for idx, item in enumerate(mlp_layer):
    X = Dense(units=item[0], activation=item[1],name='Layer'+str(idx))(X)
  A = Dense(units=action_space, activation='linear', name='Adv')(X)
  if 'DUELING' in cfg['TYPE']:
    V = Dense(units=1, activation='linear', name='Val')(X)
    Q = tf.math.add(V, A, name='Add')
    Q = tf.math.subtract(Q,tf.reduce_mean(A, axis=1, keepdims=True), name='Subtract')
  else:
    Q = A

  model = Model(inputs = IN, outputs = Q, name='mlp_network')
  model.build(input_shape = state_space)
  # dot_img_file = 'visualize_model\get_mlp_network.png'
  # plot_model(model, to_file=dot_img_file,dpi=100,
  #       show_shapes=True, show_dtype=True, show_layer_names=True,\
  # )
  return model

def get_cnn_mlp_network(state_space, action_space, cfg):
  '''
  cfg={
    'NETWORK':{
      'CNN':(
        # Name, kernel_size, strides, filters, Activation
        ('Conv2D',8,4,32,'relu'),
        ('MaxPool2D',3,1),
        ('Dropout',0.1),
        ('BN',),
        ('Conv2D',4,2,64,'relu'),
        ('AvgPool2D',3,1),
        ('Conv2D',3,1,64,'relu'),
      ),
      'MLP':(
        (64,'relu'),
        (64,'relu'),
        (64,'relu'),
      )
    },
    'TYPE':'',
  }
  model = get_cnn_mlp_network((84,84,4),4,cfg)
  model.summary()
  '''
  cnn_layer = cfg['NETWORK']["CNN"]
  mlp_layer = cfg['NETWORK']["MLP"]
  IN  = Input(shape=state_space, name='Input_Image')

  # CNN Layers
  S = IN
  for idx, item in enumerate(cnn_layer):
    if item[0] == 'Conv2D':
      S = Conv2D(filters=item[3], kernel_size=item[1], strides=item[2], padding='valid', data_format='channels_last', name='Conv-'+str(idx+1), activation=item[4])(S)
    elif item[0] == 'MaxPool2D': 
      S = MaxPool2D(              pool_size=item[1],   strides=item[2], padding="valid", data_format="channels_last", name='MaxPool2D-'+str(idx+1))(S)
    elif item[0] == 'AvgPool2D': 
      S = AveragePooling2D(       pool_size=item[1],   strides=item[2], padding="valid", data_format="channels_last", name='AvgPool2D-'+str(idx+1))(S)
    elif item[0] == 'BN':
      S = BatchNormalization(axis=-1,     name='BN-'+str(idx+1))(S)
    elif item[0] == 'Dropout':
      S = SpatialDropout2D(rate=item[1],  name='Dropout-'+str(idx+1))(S)
  S = Flatten()(S)

  # MLP Layers
  X = S
  for idx, item in enumerate(mlp_layer):
    X = Dense(units=item[0], activation=item[1],name='Layer-'+str(idx))(X)
  A = Dense(units=action_space, activation='linear', name='Adv')(X)
  if 'DUELING' in cfg['TYPE']:
    V = Dense(units=1, activation='linear', name='Val')(X)
    Q = tf.math.add(V, A, name='Add')
    Q = tf.math.subtract(Q,tf.reduce_mean(A, axis=1, keepdims=True), name='Subtract')
  else:
    Q = A

  model = Model(inputs = IN, outputs = Q, name='cnn_mlp_network')
  model.build(input_shape = state_space)
  # dot_img_file = 'visualize_model\get_cnn_mlp_network.png'
  # plot_model(model, to_file=dot_img_file,dpi=100,
  #       show_shapes=True, show_dtype=True, show_layer_names=True,\
  # )
  return model

if __name__ == "__main__":
  # cfg={
  #   'NETWORK':{
  #     'MLP':(64,64)
  #   },
  #   'TYPE':'DUELING',
  # }
  # model = get_q_network((8,),4,cfg)
  # model.summary()

  cfg={
    'NETWORK':{
      'CNN':(
        # Name, kernel_size, strides, filters, Activation
        ('Conv2D',8,4,32,'relu'),
        ('MaxPool2D',3,1),
        ('Dropout',0.1),
        ('BN',),
        ('Conv2D',4,2,64,'relu'),
        ('AvgPool2D',3,1),
        ('Conv2D',3,1,64,'relu'),
      ),
      'MLP':(
        (64,'relu'),
        (64,'relu'),
        (64,'relu'),
      )
    },
    'TYPE':'',
  }
  model = get_cnn_mlp_network((84,84,4),4,cfg)
  model.summary()

  # cfg={
  #   'NETWORK':{
  #     'MLP':(
  #       (64,'relu'),
  #       (64,'relu'),
  #       (64,'relu'),
  #     )
  #   },
  #   'TYPE':'',
  # }
  # model = get_mlp_network((8,),4,cfg)
  # model.summary()