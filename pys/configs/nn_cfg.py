from collections import OrderedDict

cfg0={
  'NETWORK':{
    'MLP': OrderedDict([
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

classic_cfg={
  'MLP': OrderedDict([
    ('NAME','Q'),
    ('STATE', OrderedDict([
      ('MLP',(
          (64,'relu'),
          (64,'relu'),
      ))
    ])),
  ])
}

model_cfg_lunarlander_continuous={
  'ACTOR': OrderedDict([
    ('NAME','ACTOR'),
    # ('ACTION_TYPE','DETERMINISTIC'),
    ('ACTION_TYPE','STOCHASTIC'),
    ('LOG_MIN_MAX',[-5,20]),
    ('STATE', OrderedDict([
      ('MLP',(
          (16,'relu'),
          (16,'relu'),
      ))
    ])),
  ]),
  'CRITIC': OrderedDict([
    ('NAME','CRITIC'),
    ('STATE', OrderedDict([
      ('MLP',(
          (16,'relu'),
          (16,'relu'),
      ))
    ])),
    ('ACTION', OrderedDict([
      ('MLP',(
          (16,'relu'),
          (16,'relu'),
      ))
    ])),
    ('MERGE', OrderedDict([
      ('MLP',(
          (16,'relu'),
          (16,'relu'),
      ))
    ]))
  ]),
}