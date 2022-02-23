from inspect import trace
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
from datetime import date
from PIL import Image
import os
from gen_and_aug import batch_size

enc = Encoder(LATENT_CHANNELS, IMAGE_SIZE)
class GAN(keras.Model):
    def __init__(self, encoder: Model, generator: Model, discriminator: Model):
        super(GAN, self).__init__()
        self.encoder = encoder
        self.generator = generator
        self.discriminator = discriminator
    
    
    def compile(self, optimizers: dict[str, Optimizer], losses: dict[str, Loss]):
        super(GAN, self).compile()
        self.e_optimizer = optimizers['encoder']
        self.g_optimizer = optimizers['generator']
        self.d_optimizer = optimizers['discriminator']
        self.b_loss = losses['BCE']
        self.l_loss = losses['L1']
        self.d_loss = losses['discriminator']
        self.enc_loss = keras.metrics.Mean(name='e_loss')
        self.gen_loss = keras.metrics.Mean(name='g_loss')
        self.disc_loss = keras.metrics.Mean(name='d_loss')
    
        
    def call(self,inputs):
        return self.generator(self.encoder(inputs))
            
    @property
    def metrics(self):
        return [self.gen_loss, self.enc_loss, self.disc_loss]
    
    def train_step(self, real_images):
        fake_images = self.generator(self.encoder(real_images))

        inp_x = [self.encoder(real_images), real_images]

        label_real = tf.random.uniform(minval=.855, maxval=.999, shape=[batch_size, 1])

        inp_x_fake = [self.encoder(real_images), fake_images]
        label_fake = tf.random.uniform(minval=0.005, maxval=.155, shape=[batch_size, 1])

        input_images = list(map(lambda x: tf.concat([x[0], x[1]], axis=0) ,zip(inp_x, inp_x_fake)))
        label = tf.concat([label_real, label_fake], axis=0)


        with tf.GradientTape() as disc_tape:
            # output = self.discriminator(input_images, training=True)
            r_out = self.discriminator(inp_x, label_real)
            # r_out = self.discriminator([tf.expand_dims(i, axis=0) for i in inp_x], tf.expand_dims(label_real, axis=0)) # HACK: Use this if the above line doesn't work
            disc_loss = self.d_loss(label, r_out)

            f_out = self.discriminator(inp_x_fake, label_fake)
            # f_out = self.discriminator([tf.expand_dims(i, axis=0) for i in inp_x_fake], tf.expand_dims(label_fake, axis=0)) # HACK: Use this if the above line doesn't work

            disc_loss += self.d_loss(label, f_out)

        disc_grads = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(disc_grads, self.discriminator.trainable_variables))

        # ------------------- Disc Training End

        label = tf.random.uniform(minval=.895, maxval=.999, shape=[batch_size, 1])

        with tf.GradientTape() as gen_tape:
            enc_img = self.encoder(real_images, training=True)
            fake_images = self.generator(enc_img, training=True)
            inp_x_fake = [enc_img, fake_images]
            output = self.discriminator(inp_x_fake, training=True)
            # print(f"{output.shape} ~ {label.shape}")
            # print(f"{real_images.shape} ~ {tf.reshape(fake_images, [None, -1])} = {self.l_loss(real_images, fake_images).shape}")
            lr_i = tf.reshape(real_images, [batch_size, -1])
            lf_i = tf.reshape(fake_images, [batch_size, -1])
            # print(lr_i.shape, lf_i.shape)
            gen_loss = self.b_loss(output, label) + 2 * self.l_loss(lr_i, lf_i)


        gen_grads = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(gen_grads, self.generator.trainable_variables))


        with tf.GradientTape() as enc_tape:
            enc_img = self.encoder(real_images, training=True)
            fake_images = self.generator(enc_img, training=True)
            inp_x_fake = [enc_img, fake_images]

            output = self.discriminator(inp_x_fake, training=True)
            lr_i = tf.reshape(real_images, [batch_size, -1])
            lf_i = tf.reshape(fake_images, [batch_size, -1])
            enc_loss = self.b_loss(output, label) + 2 * self.l_loss(lr_i, lf_i)
        
        enc_grads = enc_tape.gradient(enc_loss, self.encoder.trainable_variables)
        self.e_optimizer.apply_gradients(zip(enc_grads, self.encoder.trainable_variables))
        
        self.disc_loss.update_state(disc_loss)
        self.gen_loss.update_state(gen_loss)
        self.enc_loss.update_state(enc_loss)
        
        return {
            'g_loss': self.gen_loss.result(),
            'e_loss': self.enc_loss.result(),
            'd_loss': self.disc_loss.result(),
        }

                
            # ----------------------- DONE STEP
class GANCallBack(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        print("Screw it! I'm going to train my GAN!")
        self.model.generator.save_weights(f"gan_weights/generator_weights_{epoch}.h5")
        self.model.encoder.save_weights(f"gan_weights/encoder_weights_{epoch}.h5")
        self.model.discriminator.save_weights(f"gan_weights/discriminator_weights_{epoch}.h5")
        




        



