import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras.regularizers import l1
from tensorflow.keras.losses import Loss
import numpy as np
from tqdm import tqdm
from generator import INPUT_SHAPE as ENC_OUT_SHAPE, model as gen
from encoder import Encoder
from discriminator import IMAGE_SIZE, LATENT_CHANNELS, model as disc
from tensorflow.keras.utils import array_to_img
from datetime import datetime
from PIL import Image

enc = Encoder(LATENT_CHANNELS, IMAGE_SIZE)



class GAN:
    def __init__(self, encoder: Model, generator: Model, discriminator: Model):
        self.encoder = encoder
        self.generator = generator
        self.discriminator = discriminator
        self.compileT = False
    def compile(self, optimizers: dict[str, Optimizer], losses: dict[str, Loss]):
        self.e_optimizer = optimizers['encoder']
        self.g_optimizer = optimizers['generator']
        self.d_optimizer = optimizers['discriminator']
        self.b_loss = losses['BCE']
        self.l_loss = losses['L1']
        self.d_loss = losses['discriminator']
        self.compileT = True
    
    @tf.function
    def train_epoch(self, dataset):
        if not self.compileT:
            raise Exception('GAN not compiled')
        for step_, (real_images) in tqdm(enumerate(dataset)):

            fake_images = self.generator(self.encoder(real_images))

            inp_x = [self.encoder(real_images), real_images]
            label_real = tf.random.uniform(minval=.855, maxval=.999, shape=[real_images.shape[0], 1])

            inp_x_fake = [self.encoder(real_images), fake_images]
            label_fake = tf.random.uniform(minval=0.005, maxval=.155, shape=[real_images.shape[0], 1])

            input_images = list(map(lambda x: tf.concat([x[0], x[1]], axis=0) ,zip(inp_x, inp_x_fake)))
            label = tf.concat([label_real, label_fake], axis=0)


            with tf.GradientTape() as disc_tape:
                output = self.discriminator(input_images)
                
                disc_loss = self.d_loss(label, output)
            
            disc_grads = disc_tape.gradient(disc_loss, self.discriminator.trainable_weights)
            self.d_optimizer.apply_gradients(zip(disc_grads, self.discriminator.trainable_weights))

            # ------------------- Disc Training End

            label = tf.random.uniform(minval=.895, maxval=.999, shape=[real_images.shape[0], 1])

            with tf.GradientTape() as gen_tape:
                enc_img = self.encoder(real_images)
                fake_images = self.generator(enc_img)
                inp_x_fake = [enc_img, fake_images]
                output = self.discriminator(inp_x_fake)
                # print(f"{output.shape} ~ {label.shape}")
                # print(f"{real_images.shape} ~ {tf.reshape(fake_images, [None, -1])} = {self.l_loss(real_images, fake_images).shape}")
                lr_i = tf.reshape(real_images, [real_images.shape[0], -1])
                lf_i = tf.reshape(fake_images, [fake_images.shape[0], -1])
                # print(lr_i.shape, lf_i.shape)
                gen_loss = self.b_loss(output, label) + 2 * self.l_loss(lr_i, lf_i)


            gen_grads = gen_tape.gradient(gen_loss, self.generator.trainable_weights)
            self.g_optimizer.apply_gradients(zip(gen_grads, self.generator.trainable_weights))


            with tf.GradientTape() as enc_tape:
                enc_img = self.encoder(real_images)
                fake_images = self.generator(enc_img)
                inp_x_fake = [enc_img, fake_images]

                output = self.discriminator(inp_x_fake)
                lr_i = tf.reshape(real_images, [real_images.shape[0], -1])
                lf_i = tf.reshape(fake_images, [fake_images.shape[0], -1])
                enc_loss = self.b_loss(output, label) + 2 * self.l_loss(lr_i, lf_i)
            
            enc_grads = enc_tape.gradient(enc_loss, self.encoder.trainable_weights)
            self.e_optimizer.apply_gradients(zip(enc_grads, self.encoder.trainable_weights))

            if step_ % 100 == 0:
                
                import os
                image = os.path.join("images", "train", "000000.jpg")
                from keras.preprocessing.image import load_img, img_to_array
                
                img = self.generator(self.encoder(tf.expand_dims(img_to_array(load_img(image, target_size=[256, 256])), 0)))
                img *= 255
                img.numpy()
                image = array_to_img(tf.squeeze(img))
                image.save(f'./results/Epoch-{step_}-{str(datetime.now())}')

            # ----------------------- DONE STEP





        



