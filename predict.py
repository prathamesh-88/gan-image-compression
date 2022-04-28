
# from gan import encoder, generator, discriminator, generator_optimizer, encoder_optimizer, discriminator_optimizer
from generator import model as generator
from discriminator import model as discriminator, LATENT_CHANNELS, IMAGE_SIZE
from encoder import build_encoder
encoder = build_encoder(LATENT_CHANNELS, (None, *IMAGE_SIZE))

from tensorflow.keras.optimizers import Adam
generator_optimizer, encoder_optimizer, discriminator_optimizer = Adam(1e-4), Adam(1e-4), Adam(1e-4)

import tensorflow as tf
from gen_and_aug import preprocess, postprocess
from tensorflow.keras.preprocessing.image import array_to_img, load_img, img_to_array
import os
import numpy as np

DEBUG = False

CHECKPOINT_DIR = os.path.join(".", "training_checkpoints")
CHECKPOINT_PREFIX = os.path.join(CHECKPOINT_DIR, "ckpt")
checkpoint = tf.train.Checkpoint(
    generator_optimizer= generator_optimizer,
    discriminator_optimizer= discriminator_optimizer,
    encoder_optimizer= encoder_optimizer,
    encoder= encoder, generator= generator,
    discriminator= discriminator
)

manager = tf.train.CheckpointManager(checkpoint, CHECKPOINT_DIR, max_to_keep=3)
checkpoint.restore(tf.train.latest_checkpoint(CHECKPOINT_DIR))


print(manager.latest_checkpoint) if DEBUG else None


def generate_latent_space(image_tensor):
    image_tensor = preprocess(image_tensor)
    its = image_tensor.shape
    print(its) if DEBUG else None
    if len(its) == 3:
        return tf.squeeze(encoder(tf.expand_dims(image_tensor, axis=0)), axis=0)
    else:
        return encoder(image_tensor)

def generate_image(latent_tensor):
    lts = latent_tensor.shape
    print(lts) if DEBUG else None

    if len(lts) == 3:
        return postprocess(tf.squeeze(generator(tf.expand_dims(latent_tensor, axis=0)), axis=0))
    else:
        return postprocess(generator(latent_tensor))


def generate_latent_image(latent_tensor, grayscale=True, debug=True):
    if len(latent_tensor.shape) == 3:
        if not grayscale:
            images = [latent_tensor[:, :, i:i+3] for i in range(0, latent_tensor.shape[-1], 3)]
            images = [np.hstack(images[i:i+3]) for i in range(0, len(images), 3)]
            images = np.vstack(images)
            img = array_to_img(images)
            if debug:
                img.save("debug_.png")
            else:
                return array_to_img(images)

        else:
            images = [latent_tensor[:, :, i:i+1] for i in range(0, latent_tensor.shape[-1])]
            images = [np.vstack(images[i:i+3]) for i in range(0, len(images), 3)]
            images = np.hstack(images)
            img = array_to_img(images)
            if debug:
                img.save("debug_g.png")
            else:
                return img

def div_images(image):
    # Split the image into 9 regions and return the list of the regions
    image = img_to_array(image)
    print(image.shape)
    h, w = image.shape[:2]
    s = int(np.floor(h/3)), int(np.floor(w/3))
    # return the i
    return [image[i*s[0]:(i+1)*s[0], j*s[1]:(j+1)*s[1], :] for i in range(3) for j in range(3)]

def merge_list_over_axis(img):
    # merge the list over axis -1
    print([i.shape for i in img])
    return np.concatenate(img, axis=-1)
        


if __name__ == "__main__":
    import argparse
    args = argparse.ArgumentParser(description="CLI Tool for Image Compression and Decompression")
    args.add_argument('input', metavar='input', type=str, help='Input File to perform action')
    args.add_argument('-c', '--compress',
                        action='store_true', help="compress the provided file")
    args.add_argument('-d', '--decompress',
                        action='store_true', help="decompress the provided file")
    args.add_argument('-o', '--output', required=True, action='store', help='path to output file')

    args.add_argument('-r', '--replicate', action='store_true', help='generates final form of image')
    args.add_argument('-v', '--verbose', action='store_true', help='verbose output')
    
    arguments = args.parse_args()

    input_file: str = arguments.input
    verbose = arguments.verbose
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"{input_file} not found!")

    if arguments.replicate:
        image = load_img(input_file)
        image = img_to_array(image)
        data = generate_latent_space(image)
        data = postprocess(data)
        data = tf.cast(data, tf.uint8)
        generate_latent_image(data) if verbose else None
        generate_latent_image(data, False) if verbose else None
        data = tf.cast(data, tf.float32)
        data = preprocess(data)
        image = generate_image(data)
        image = array_to_img(image)
        image.save(arguments.output)
        exit(0)

    
    c = arguments.compress
    d = arguments.decompress
    if ((not c) and (not d)) or (c and d):
        raise TypeError("both operations absent or present")
    
    if d:
        img = load_img(input_file)
        new = div_images(img)
        new = merge_list_over_axis(new)
        print(new.shape)
        data = tf.cast(new, tf.float32)
        data = preprocess(data)
        image = generate_image(data)
        image = array_to_img(image)
        image.save(arguments.output)
    else:
        image = load_img(input_file)
        image = img_to_array(image)
        data = generate_latent_space(image)
        data = postprocess(data)
        data = tf.cast(data, tf.uint8)
        img = generate_latent_image(data, grayscale=False, debug=False)
        img.save(arguments.output)
        
    
    
