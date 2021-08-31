import gym
import cv2
import numpy as np
from gym import spaces

class GymWrapper():
    def __init__(self,cfg):
        self.env_name   = cfg['NAME']
        self.img_size   = cfg['IMG_SIZE']
        self.img_crop   = cfg['IMG_CROP']
        self.state_type = cfg["STATE_TYPE"]
        self.img_type   = cfg['IMG_TYPE']
        self.env = gym.make(self.env_name)
        self.observation_space  = self.env.observation_space
        self.action_space       = self.env.action_space
        if self.state_type == "IMG":
            self.img_data_type = np.uint8
            self.observation_space  = spaces.Box(low=0,high=1, shape=self.img_size, dtype=self.img_data_type)

        return

    def reset(self):
        state = self.env.reset()
        if self.state_type == "IMG":
            raw_img = self.env.render(mode='rgb_array')
            state = self._preprocessing_img(raw_img=raw_img)
        return state

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        
        if self.state_type == "IMG":
            raw_img = self.env.render(mode='rgb_array')
            next_state = self._preprocessing_img(raw_img=raw_img)
        return next_state, reward, done, info

    def render(self):
        raw_img = self.env.render(mode='human')
        # cv2.waitkey(0)
        return

    def close(self)->None:
        self.env.close()
        return
        
    def _preprocessing_img(self, raw_img):
        # print(np.shape(raw_img))
        img_rgb_crip = raw_img[self.img_crop[0][0]:self.img_crop[0][1],self.img_crop[1][0]:self.img_crop[1][1],:]
        # print(np.shape(img_rgb_crip))
        img_rgb_resize = cv2.resize(img_rgb_crip, self.img_size[0:2], interpolation=cv2.INTER_CUBIC)
        img_rgb_resize = np.transpose(img_rgb_resize,axes=(1,0,2))
        # print(np.shape(img_rgb_resize))
        if self.img_type == 'GRAY':
            img_k_resize = cv2.cvtColor(img_rgb_resize,cv2.COLOR_RGB2GRAY)
            img_k_resize = img_k_resize / 255.0 # scaling 0 ~ 1
            # img_k_resize = img_k_resize / 127.5 - 1. # scaling -1 ~ 1
            preprocessed_state = np.array(img_k_resize,dtype=self.img_data_type)
            preprocessed_state = np.expand_dims(preprocessed_state,axis=2)
        elif self.img_type == 'RGB':
            preprocessed_state = img_rgb_resize

        return preprocessed_state