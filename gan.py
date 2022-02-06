import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras.losses import Loss
import numpy as np
from generator import INPUT_SHAPE as ENC_OUT_SHAPE, model as gen
from encoder import Encoder
from discriminator import IMAGE_SIZE, LATENT_CHANNELS, model as disc


enc = Encoder(LATENT_CHANNELS, IMAGE_SIZE)


class GAN:
    def __init__(self, encoder: Model, generator: Model, discriminator: Model):
        self.encoder = encoder
        self.generator = generator
        self.discriminator = discriminator
        self.compile = False
    def compile(self, optimizers: dict[str, Optimizer], losses: dict[str, Loss]):
        self.e_optimizer = optimizers['encoder']
        self.g_optimizer = optimizers['generator']
        self.d_optimizer = optimizers['discriminator']
        self.b_loss = losses['BCE']
        self.l_loss = losses['L1']
        self.d_loss = losses['discriminator']
        self.compile = True
    
    @tf.function
    def train_epoch(self, dataset):
        if not compile:
            raise Exception('GAN not compiled')
        for step, (real_images) in enumerate(dataset):

            fake_images = self.generator(self.encoder(real_images))

            inp_x = [self.encoder(real_images), real_images]
            label_real = tf.random.uniform(minval=.855, maxval=.999, shape=[real_images.shape[0], 1])

            inp_x_fake = [self.encoder(real_images), fake_images]
            label_fake = tf.random.uniform(minval=0.005, maxval=.155, shape=[real_images.shape[0], 1])

            input_images = list(map(lambda x, y: tf.concat([x, y], axis=0) ,zip(inp_x, inp_x_fake)))
            label = tf.concat([label_real, label_fake], axis=0)


            with tf.GradientTape() as disc_tape:
                output = self.discriminator(input_images)
                
                disc_loss = self.d_loss(label, output)
            
            grads = disc_tape.gradient(disc_loss, self.discriminator.trainable_weights)
            self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

            # ------------------- Disc Training End

            label = tf.random.uniform(minval=.895, maxval=.999, shape=[real_images.shape[0], 1])

            with tf.GradientTape() as gen_tape:
                enc_img = self.encoder(real_images)
                fake_images = self.generator(enc_img)
                inp_x_fake = [enc_img, fake_images]
                output = self.discriminator(inp_x_fake)

                gen_loss = self.b_loss(output, label) + 2 * self.l_loss(real_images, fake_images)
                






        



