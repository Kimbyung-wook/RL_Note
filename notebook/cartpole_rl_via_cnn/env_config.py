env_configs = {
    # Discrete Action Env
    'CartPole-v1':{
        'EPISODES':6000,
        'END_SCORE':400
    },
    'LunarLander-v2':{
        'EPISODES':1000,
        'END_SCORE':300
    },
    'MountainCar-v0':{
        'EPISODES':2000,
        'END_SCORE':-100
    },
    # Continuous Action Env
    'Pendulum-v0':{
        'EPISODES':500,
        'END_SCORE':-200
    },
    'LunarLanderContinuous-v2':{
        'EPISODES':2000,
        'END_SCORE':200
    },
    'MountainCarContinuous-v0':{
        'EPISODES':5000,
        'END_SCORE':100
    }
}