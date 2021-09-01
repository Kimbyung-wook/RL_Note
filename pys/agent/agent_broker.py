from pys.agent.dqn_agent  import DQNAgent
from pys.agent.mdqn_agent import MDQNAgent

from pys.agent.ddpg_agent import DDPGAgent
from pys.agent.td3_agent  import TD3Agent
from pys.agent.sac_agent  import SACAgent

def agent_broker(rl:str, env, cfg):
    if   rl == "DQN":   agent = DQNAgent(env, cfg)
    elif rl == "A2C":   agent = A2CAgent(env, cfg)
    elif rl == "MDQN":  agent = MDQNAgent(env, cfg)

    elif rl == "DDPG":  agent = DDPGAgent(env, cfg)
    elif rl == "TD3":   agent = TD3Agent(env, cfg)
    elif rl == "SAC":   agent = SACAgent(env, cfg)
    else:
        Exception(rl + " is not exist")

    return agent
