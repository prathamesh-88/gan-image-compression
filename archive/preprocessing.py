import numpy as np
from PIL import Image

class Resizer:
    
    def __init__(self,size=(500,500)):
        self.size   = size
        self.result = []

    # To get the padding part of the image. It inverts the actual RGB value of the image    
    def get_padding_bits(self,img_arr:Image, size:tuple):
        mean_color = np.mean(img_arr, axis=(0, 1))
        mean_color = np.array([mean_color]*(size[0]*size[1]), dtype='uint8').reshape(size[0],size[1],3)
        return 255-mean_color

    
    # To display the result 
    def show_result(self):
        if not len(self.result):
            print("There is no result to show! Please run preprocess.fit first")
        else:
            Image.fromarray(self.result).show()

    # Actual function
    def fit(self, img_path):
        image = Image.open(img_path)
        mode  = (0,1) [image.size[1] > image.size[0]]
        a_ratio = image.size[0]/image.size[1]

        
        if mode:
            adj_x  = int(a_ratio * self.size[1])
            mod_size = (adj_x, self.size[1])
            res      = image.resize(mod_size)
            padding_image = self.get_padding_bits(image,(self.size[1], self.size[0]-adj_x))
            img_array = np.array(res, dtype=np.uint8)
            res = np.hstack([img_array, padding_image])
        
        else:
            adj_y = int(self.size[0]/a_ratio)
            mod_size = ( self.size[0], adj_y)
            res     = image.resize(mod_size)
            padding_image = self.get_padding_bits(image, (self.size[1]-adj_y, self.size[0]))
            img_array = np.array(res, dtype=np.uint8)
            res = np.vstack([img_array, padding_image])

        self.result = res

if __name__ == '__main__':
    obj = Resizer()
    obj.fit('test_images/sample0.jpeg')
    obj.show_result()