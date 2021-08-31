import cv2
import numpy as np

class Featurization():
  def __init__(self, observation_size):
    self.obs_size = observation_size
    self.is_first = True
    self.feature = np.zeros(observation_size)

  def preprocessing(self, img):
    img_rgb_resize = cv2.resize(img, self.obs_size[0:2], interpolation=cv2.INTER_CUBIC)
    img_rgb_resize = np.transpose(img_rgb_resize,axes=(1,0,2))
    img_k_resize = cv2.cvtColor(img_rgb_resize,cv2.COLOR_RGB2GRAY)
    # img_k_resize = img_k_resize / 255.0 # scaling 0 ~ 1
    state = np.array(img_k_resize,dtype=np.uint8)
    state = np.expand_dims(state,axis=2)
    if self.is_first == True:
      for i in range(self.obs_size[2]):
        self.feature = np.append(self.feature, state, axis=2)
        self.feature = np.delete(self.feature, obj=0, axis=2)
      self.is_first = False
    else:
      self.feature = np.append(self.feature, state, axis=2)
      self.feature = np.delete(self.feature, obj=0, axis=2)
    return self.feature