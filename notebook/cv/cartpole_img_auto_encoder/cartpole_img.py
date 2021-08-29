import gym
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2

env = gym.make("CartPole-v1")
env.reset()
img_size = (128,96)

def get_image(img_rgb):
    img_rgb_resize = cv2.resize(img_rgb, (img_size[0],img_size[1]), interpolation=cv2.INTER_CUBIC)
    img_k_resize = cv2.cvtColor(img_rgb_resize,cv2.COLOR_RGB2GRAY)
    # img_k_resize = img_k_resize / 255.0 # scaling 0 ~ 1
    # img_k_resize = img_k_resize / 127.5 - 1. # scaling -1 ~ 1
    state = img_k_resize
    return state
    
imgs = get_image(env.render(mode='rgb_array'))
imgs = np.array([imgs],dtype=np.uint8)
print('img : ',np.shape(imgs), type(imgs))

for i in range(50000-1):
    state = get_image(env.render(mode='rgb_array'))
    state = np.array([state],dtype=np.uint8)
    # print(np.shape(state), type(state))
    imgs = np.append(imgs, state, axis=0)
    # state = np.expand_dims(state,axis=2)
    # cv2.imwrite("cartpole_img\\cartpole_img{:05d}.jpg".format(i),state[0])
    _, _, done, _ = env.step(random.randrange(env.action_space.n))
    if done == True:
        env.reset()
    if i % 1000 == 0:
        print('Save imgs shape : ',np.shape(imgs))
        with open('cartpole_img.npy','wb') as fid:
            np.save(fid, imgs)