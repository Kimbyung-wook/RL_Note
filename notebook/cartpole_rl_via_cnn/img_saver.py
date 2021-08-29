import random
import numpy as np
import cv2

class ImgSaver():
    def __init__(self, image_size=(128,96), save_name='tmp.npy'):
        self.first = True
        self.image_size = image_size
        self.imgs = None
        self.idx = 0
        self.name = save_name
        pass

    def get_image_to_gray(self, img_rgb):
        self.idx += 1
        img_rgb_resize = cv2.resize(img_rgb, (self.image_size[0],self.image_size[1]), interpolation=cv2.INTER_CUBIC)
        img_k_resize = cv2.cvtColor(img_rgb_resize,cv2.COLOR_RGB2GRAY)
        # img_k_resize = img_k_resize / 255.0 # scaling 0 ~ 1
        # img_k_resize = img_k_resize / 127.5 - 1. # scaling -1 ~ 1
        state = img_k_resize
        state = np.array([state],dtype=np.uint8)
        if self.first == True:
            self.first = False
            self.imgs = state
        else:
            self.imgs = np.append(self.imgs, state, axis=0)

        if self.idx % 1000 == 0:
            self.save_data()
            print('Image idx : ', self.idx, np.shape(self.imgs))

    def save_data(self):
        with open(self.name,'wb') as fid:
            np.save(fid, self.imgs)
            