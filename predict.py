
from gan import get_models
import tensorflow as tf

from tensorflow.keras.preprocessing.image import array_to_img, load_img, img_to_array

encoder, generator = get_models()


def generate_latent_space(image_tensor):
    if len(image_tensor.shape) == 3:
        image_tensor = tf.expand_dims(image_tensor, axis=0)
    
    data = encoder(image_tensor)
    if len(image_tensor.shape) == 3:
        return data[0]
    else:
        return data

def generate_image(latent_tensor):
    if len(latent_tensor.shape) == 3:
        latent_tensor = tf.expand_dims(latent_tensor, axis=0)
    data = generator(latent_tensor)
    if len(latent_tensor.shape) == 3:
        return data[0]
    else:
        return data




