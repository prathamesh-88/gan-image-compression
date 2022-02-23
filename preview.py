from datetime import date
import numpy as np
import tensorflow as tf
from gen_and_aug import datagen
from tensorflow.keras.utils import array_to_img
# tf.config.run_functions_eagerly(True)
from discriminator import IMAGE_SIZE, LATENT_CHANNELS
from gan import enc, gen, IMAGE_SIZE


real_images = datagen("./images", IMAGE_SIZE)[0]
    # enc = Encoder(LATENT_CHANNELS, IMAGE_SIZE)
    # enc.load_weights('gan_weights/encoder_weights.h5')
    # gen = Generator()
    # gen.load_weights('gan_weights/generator_weights.h5')

for i in range(20):
    sample_image = real_images[:1]
    print(f'Sample Image: {sample_image.shape}')
    encoded_sample_image = enc(sample_image)
    enc.load_weights(f'gan_weights/encoder_weights_{i}.h5')
    encoded_sample_image = enc(sample_image)

    # print(f'Encoded Image: {encoded_sample_image.shape}')
    generated_sample_image = gen(encoded_sample_image)[0]
    gen.load_weights(f'gan_weights/generator_weights_{i}.h5')
    generated_sample_image = gen(encoded_sample_image)[0]

    generated_sample_image *= 255
    real_image = real_images[0] *255
    # print(f'Generated Image: {generated_sample_image.shape}')
    # print(f'Get Image: {generated_sample_image[0].shape}')
    compare = np.hstack([real_image, generated_sample_image])
    # print(type(real_images[0]))
    name = str(i)
    array_to_img(compare).save(f'results/{str(date.today())}_{name}_generated_image.png')