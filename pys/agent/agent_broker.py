from pys.agent.dqn_agent  import *
from pys.agent.mdqn_agent import *
from pys.agent.ddpg_agent import *
from pys.agent.td3_agent  import *
from pys.agent.sac_agent  import *

def discrete_agent_broker(rl:str, env, cfg):
  print('length ',len(cfg['ENV']['STATE']['TYPE']))
  agent = None
  if   len(cfg['ENV']['STATE']['TYPE']) == 1:
    if   rl == "DQN":   agent = DQNAgent1( env, cfg)
    elif rl == "A2C":   agent = A2CAgent1( env, cfg)
    elif rl == "MDQN":  agent = MDQNAgent1(env, cfg)
    else:
        Exception(rl + " is not exist")
  elif len(cfg['ENV']['STATE']['TYPE']) == 2:
    if   rl == "DQN":   agent = DQNAgent2( env, cfg)
    elif rl == "A2C":   agent = A2CAgent2( env, cfg)
    elif rl == "MDQN":  agent = MDQNAgent2(env, cfg)
    else:
        Exception(rl + " is not exist")
  print('load : ',agent.get_name())

  return agent
def continuous_agent_broker(rl:str, env, cfg):
  print('length ',len(cfg['ENV']['STATE']['TYPE']))
  agent = None
  if   len(cfg['ENV']['STATE']['TYPE']) == 1:
    if   rl == "DDPG":  agent = DDPGAgent1(env, cfg)
    elif rl == "TD3":   agent = TD3Agent1( env, cfg)
    elif rl == "SAC":   agent = SACAgent1( env, cfg)
    else:
      Exception(rl + " is not exist")
  elif len(cfg['ENV']['STATE']['TYPE']) == 2:
    if   rl == "DDPG":  agent = DDPGAgent2(env, cfg)
    elif rl == "TD3":   agent = TD3Agent2( env, cfg)
    elif rl == "SAC":   agent = SACAgent2( env, cfg)
    else:
      Exception(rl + " is not exist")

  return agent
