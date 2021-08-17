import numpy as np

def pendulum_done(state):
    return False
    
def pendulum_reward(state, action, reward_old, next_state, done):
    costh   = state[0]
    sinth   = state[1]
    th      = np.arctan2(sinth,costh)
    thdot   = state[2]
    u       = action[0]
    costs   = th ** 2 + 0.1 * thdot + 0.001 * (u ** 2)
    return -costs

def lunarlandercontinuous_done(state):
    return False

def lunarlandercontinuous_reward(state, action, reward_old, next_state, done):
    m_power = action[0]
    s_power = action[1]
    reward = 0
    shaping = (
        -100 * np.sqrt(state[0] * state[0] + state[1] * state[1])
        - 100 * np.sqrt(state[2] * state[2] + state[3] * state[3])
        - 100 * abs(state[4])
        + 10 * state[6]
        + 10 * state[7]
    )  # And ten points for legs contact, the idea is if you
    # lose contact again after landing, you get negative reward
    prev_shaping = reward_old
    reward = shaping - prev_shaping

    reward -= (
        m_power * 0.30
    )  # less fuel spent is better, about -30 for heuristic landing
    reward -= s_power * 0.03

    return reward