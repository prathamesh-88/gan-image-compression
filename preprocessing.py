
import numpy as np
from PIL import Image
class preprocess:
    
    def __init__(self,size=(500,500)):
        self.size = size
        
    def fit(self, img_path):
        image = Image.open(img_path)
        pad_color = int(256 - np.mean(np.array(image))) 
        mode  = (0,1) [image.size[1] > image.size[0]]
        a_ratio = image.size[0]/image.size[1]

        
        if mode:
            adj_x = int(a_ratio * self.size[1])
            mod_size = (adj_x, self.size[1])
            print(mod_size)
            res      = image.resize(mod_size)
            padding_image = np.zeros([self.size[1], self.size[0]-adj_x,  3], dtype=np.uint8)
            padding_image.fill(pad_color)
            img_array = np.array(res)
            print(padding_image.shape)
            print(img_array.shape)
            res = np.hstack([img_array, padding_image])
        
        else:
            adj_y = int(self.size[0]/a_ratio)
            mod_size = ( self.size[0], adj_y)
            res     = image.resize(mod_size)
            padding_image = np.zeros([self.size[1]-adj_y, self.size[0],  3], dtype=np.uint8)
            padding_image.fill(pad_color)
            img_array = np.array(res)
            print(padding_image.shape)
            print(img_array.shape)
            res = np.vstack([img_array, padding_image])

        return res