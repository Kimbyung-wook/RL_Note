import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Concatenate, concatenate
from tensorflow.keras.layers import Conv2D, Flatten, MaxPool2D, BatchNormalization, AveragePooling2D, SpatialDropout2D
from tensorflow.keras.utils  import plot_model
from collections import OrderedDict
import numpy as np

def network_maker1(state_space, action_space, cfg):
  structure   = cfg
  role        = cfg['NAME']
  S      = Input(shape=state_space, name='Input_Image')
  A      = Input(shape=action_space,name='Input_Action')
  if    role == 'ACTOR' or role == 'Q': IN  = S
  elif  role == 'CRITIC':               IN  = [S, A]

  for key, item in structure.items():
    if key == 'NAME':
      continue
    elif key == 'STATE':
      S = get_layers(S, key, item)
    elif key == 'ACTION':
      A = get_layers(A, key, item)
    elif key == 'MERGE':
      if role == 'ACTOR' or role == 'Q':
        M  = S
      elif role == 'CRITIC':
        M  = concatenate([S, A],axis=1)
      M = get_layers(M, key, item)
  if 'MERGE' not in structure:
    if role == 'ACTOR' or role == 'Q':
      M  = S
    elif role == 'CRITIC':
      M  = concatenate([S, A],axis=1)
  if    role == 'Q':      OUT = Dense(units=action_space[0],  activation='linear',name='Output')(M)
  elif  role == 'CRITIC': OUT = Dense(units=1,                activation='linear',name='Output')(M)
  elif  role == 'ACTOR':
    if    cfg['ACTION_TYPE'] == 'DETERMINISTIC':
      MU      = Dense(units=action_space[0], activation='linear',name='MU' )(M)
      OUT     = MU
    elif  cfg['ACTION_TYPE'] == 'STOCHASTIC':
      MU      = Dense(units=action_space[0], activation='linear',name='MU' )(M)
      LOG_STD = Dense(units=action_space[0], activation='linear',name='STD')(M)
      LOG_STD = tf.clip_by_value(LOG_STD, clip_value_min=cfg['LOG_MIN_MAX'][0], clip_value_max=cfg['LOG_MIN_MAX'][1], name='CLIPPING')
      STD     = tf.math.exp(LOG_STD, name='EXP')
      OUT = [MU, STD]
    
  model = Model(inputs = IN, outputs = OUT, name=role)
  model.build(input_shape = state_space)
  return model

def network_maker2(state_space, action_space, cfg):
  structure   = cfg
  role        = cfg['NAME']
  S1     = Input(shape=state_space[0], name='Input_Image')
  S2     = Input(shape=state_space[1], name='Input_Value')
  A      = Input(shape=action_space,   name='Input_Action')
  if    role == 'ACTOR' or role == 'Q': IN  = [S1, S2]
  elif  role == 'CRITIC':               IN  = [S1, S2, A]

  for key, item in structure.items():
    if key == 'NAME':
      continue
    elif key == 'STATE1':
      S1 = get_layers(S1, key, item)
    elif key == 'STATE2':
      S2 = get_layers(S2, key, item)
    elif key == 'ACTION':
      A  = get_layers(A, key, item)
    elif key == 'MERGE':
      if role == 'ACTOR':
        M  = concatenate([S1, S2],axis=1)
      elif role == 'CRITIC':
        M  = concatenate([S1, S2, A],axis=1)
      M = get_layers(M, key, item)
  if    role == 'Q':      OUT = Dense(units=action_space[0],  activation='linear',name='Output')(M)
  elif  role == 'CRITIC': OUT = Dense(units=1,                activation='linear',name='Output')(M)
  elif  role == 'ACTOR':
    if    cfg['ACTION_TYPE'] == 'DETERMINISTIC':
      MU      = Dense(units=action_space[0], activation='linear',name='MU' )(M)
      OUT     = MU
    elif  cfg['ACTION_TYPE'] == 'STOCHASTIC':
      MU      = Dense(units=action_space[0], activation='linear',name='MU' )(M)
      LOG_STD = Dense(units=action_space[0], activation='linear',name='STD')(M)
      LOG_STD = tf.clip_by_value(LOG_STD, clip_value_min=cfg['LOG_MIN_MAX'][0], clip_value_max=cfg['LOG_MIN_MAX'][1], name='CLIPPING')
      STD     = tf.math.exp(LOG_STD, name='EXP')
      OUT = [MU, STD]
    
  model = Model(inputs = IN, outputs = OUT, name=role)
  model.build(input_shape = state_space)
  return model

def get_layers(INPUT, layer_role, cfg):
  structure = cfg
  X = INPUT
  for key, item1 in structure.items():
    if key == 'CNN':
      for idx, item2 in enumerate(item1):
        if item2[0] == 'Conv2D':
          X = Conv2D(filters=item2[3],kernel_size=item2[1], strides=item2[2], padding='valid', data_format='channels_last', name=layer_role+'-Conv-'+str(idx+1), activation=item2[4])(X)
        elif item2[0] == 'MaxPool2D': 
          X = MaxPool2D(              pool_size=item2[1],   strides=item2[2], padding="valid", data_format="channels_last", name=layer_role+'-MaxPool2D-'+str(idx+1))(X)
        elif item2[0] == 'AvgPool2D': 
          X = AveragePooling2D(       pool_size=item2[1],   strides=item2[2], padding="valid", data_format="channels_last", name=layer_role+'-AvgPool2D-'+str(idx+1))(X)
        elif item2[0] == 'BN':
          X = BatchNormalization(axis=-1,       name=layer_role+'-BN-'+str(idx+1))(X)
        elif item2[0] == 'Dropout':
          X = SpatialDropout2D(rate=item2[1],   name=layer_role+'-Dropout-'+str(idx+1))(X)
    elif key == 'FLAT':
      X = Flatten(name=key)(X)
    elif key == 'MLP':
      for idx, item2 in enumerate(item1):
        X = Dense(units=item2[0], activation=item2[1],name=layer_role+'-Layer-'+str(idx))(X)
  return X

if __name__ == '__main__':

  cfg0={
    'NETWORK':{
      'Q': OrderedDict([
        ('NAME','Q'),
        ('STATE', OrderedDict([
          ('MLP',(
              (64,'relu'),
              (64,'relu'),
          ))
        ])),
      ])
    }
  }
  cfg1={
    'NETWORK':{
      'Q': OrderedDict([
        ('NAME','Q'),
        ('STATE', OrderedDict([
          ('CNN',(
              ('Conv2D',8,4,32,'relu'),
              ('Conv2D',4,2,64,'relu'),
              ('Conv2D',3,1,64,'relu')
              # ('MaxPool2D',3,1),
              # ('Dropout',0.1)
          ),),
          ('FLAT',
            0
          ),
          ('MLP',(
              (64,'relu'),
              (64,'relu'),
          ))
        ])),
        ('MERGE', OrderedDict([
          ('MLP',(
              (64,'relu'),
              (64,'relu'),
          ))
        ]))
      ])
    }
  }
  cfg2={
    'NETWORK':{
      'ACTOR': OrderedDict([
        ('NAME','ACTOR'),
        # ('ACTION_TYPE','DETERMINISTIC'),
        ('ACTION_TYPE','STOCHASTIC'),
        ('LOG_MIN_MAX',[-5,20]),
        ('STATE1', OrderedDict([
          ('CNN',(
              ('Conv2D',8,4,32,'relu'),
              ('Conv2D',4,2,64,'relu'),
              ('Conv2D',3,1,64,'relu')
              # ('MaxPool2D',3,1),
              # ('Dropout',0.1)
          ),),
          ('FLAT',
            0
          ),
          ('MLP',(
              (64,'relu'),
              (64,'relu'),
          ))
        ])),
        ('STATE2', OrderedDict([
          ('MLP',(
              (64,'relu'),
              (64,'relu'),
          ))
        ])),
        ('MERGE', OrderedDict([
          ('MLP',(
              (64,'relu'),
              (64,'relu'),
          ))
        ]))
      ]),
      'CRITIC': OrderedDict([
        ('NAME','CRITIC'),
        ('STATE1', OrderedDict([
          ('CNN',(
              ('Conv2D',8,4,32,'relu'),
              ('MaxPool2D',3,1),
              ('Dropout',0.1)
          ),),
          ('FLAT',
            0
          ),
          ('MLP',(
              (64,'relu'),
              (64,'relu'),
          ))
        ])),
        ('STATE2', OrderedDict([
          ('MLP',(
              (64,'relu'),
              (64,'relu'),
          ))
        ])),
        ('ACTION', OrderedDict([
          ('MLP',(
              (64,'relu'),
              (64,'relu'),
          ))
        ])),
        ('MERGE', OrderedDict([
          ('MLP',(
              (64,'relu'),
              (64,'relu'),
          ))
        ]))
      ]),
    }
  }
  # state_space = (8,)
  # action_space = (4,)
  # model = network_maker1(state_space, action_space, cfg0['NETWORK']['Q'])
  # model.summary()
  # dot_img_file = 'graph_img/' + cfg0['NETWORK']['Q']['NAME'] + 'mlp.png'
  # plot_model(model, to_file=dot_img_file,dpi=100,
  #       show_shapes=True, show_dtype=True, show_layer_names=True,\
  # )

  # state_space = (64,64,4)
  # action_space = (4,)
  # model = network_maker1(state_space, action_space, cfg1['NETWORK']['Q'])
  # model.summary()
  # dot_img_file = 'graph_img/' + cfg1['NETWORK']['Q']['NAME'] + 'cnn.png'
  # plot_model(model, to_file=dot_img_file,dpi=100,
  #       show_shapes=True, show_dtype=True, show_layer_names=True,\
  # )

  state_space = ((64,64,4),4)
  action_space = (4,)
  model = network_maker2(state_space, action_space, cfg2['NETWORK']['ACTOR'])
  model.summary()
  dot_img_file = 'graph_img/' + cfg2['NETWORK']['ACTOR']['NAME'] + '2.png'
  plot_model(model, to_file=dot_img_file,dpi=100,
        show_shapes=True, show_dtype=True, show_layer_names=True,\
  )
  model = network_maker2(state_space, action_space, cfg2['NETWORK']['CRITIC'])
  model.summary()
  dot_img_file = 'graph_img/' + cfg2['NETWORK']['CRITIC']['NAME'] + '2.png'
  plot_model(model, to_file=dot_img_file,dpi=100,
        show_shapes=True, show_dtype=True, show_layer_names=True,\
  )
