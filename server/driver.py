# def compress(x):
#     return x

# def decompress(x):
#     return x

import sys, os
os.chdir("../")
sys.path.append("./")
from predict import generate_latent_space, generate_image, generate_latent_image, img_to_array, array_to_img
os.chdir("server")

def compress(x):
    x = img_to_array(x)
    return generate_latent_space(x)

def decompress(x):
    return array_to_img(generate_image(x))
