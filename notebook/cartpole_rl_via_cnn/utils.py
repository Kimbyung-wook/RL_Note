import numpy as np

class ImageFeaturization():
    def __init__(self, data_format:str = "last", img_size = (128,96,4)):
        self.data_format = data_format.upper()
        self.image_channel = (0 if self.data_format == "first" else 2)
        self.img_size = img_size
        self.feature = np.zeros((self.img_size[0], self.img_size[1], 4))
        # print('feature ',np.shape(self.feature))
        self.is_first = True
        self.idx = 0
        return

    def __call__(self, img):
        if self.is_first == True:
            self.is_first = False
            self.feature = img
            self.idx = 1
        else:
            self.feature = np.append(self.feature, img,axis=2)
            self.idx += 1
            if(self.idx > 4):
                self.feature = np.delete(self.feature, obj=0, axis=self.image_channel)
        
        if self.idx == 4:
            return self.feature, True
        else:
            return np.zeros((self.img_size[0], self.img_size[1], 4)), False

    # def attach(self, img):
    #     if self.is_first == True:
    #         self.is_first = False
    #         self.feature = img
    #         self.idx = 1
    #     else:
    #         self.feature = np.append(img,axis=2)
    #         self.idx += 1
    #         if(self.idx > 4):
    #             self.feature = np.delete(self.feature, obj=0, axis=self.image_channel)
    
    # def can_get_feature(self):
    #     return (self.idx == 4)

    # def get_feature(self):
    #     return self.feature
    
    def reset(self):
        self.feature = None
        self.is_first = True
        self.idx = 0

