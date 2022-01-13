import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model, Sequential
import os

from tensorflow.keras.utils import plot_model
import numpy as np

tile_size = (128, 128, 3) # (height, width, channels)

# Models
real_image_input = keras.Input(shape=tile_size)
conv_ri = layers.Conv2D(64, kernel_size=4, strides=2, padding="same")(real_image_input)
leaky_ri = layers.LeakyReLU(alpha=0.2)(conv_ri)
conv_ri = layers.Conv2D(128, kernel_size=4, strides=2, padding="same")(leaky_ri)
leaky_ri = layers.LeakyReLU(alpha=0.2)(conv_ri)
fake_image_input = keras.Input(shape=tile_size)
conv_fi = layers.Conv2D(64, kernel_size=4, strides=2, padding="same")(fake_image_input)
leaky_fi = layers.LeakyReLU(alpha=0.2)(conv_fi)
conv_fi = layers.Conv2D(128, kernel_size=4, strides=2, padding="same")(leaky_fi)
leaky_fi = layers.LeakyReLU(alpha=0.2)(conv_fi)
concat = layers.Concatenate()([leaky_ri, leaky_fi])
conv_c = layers.Conv2D(128, kernel_size=3, padding="same")(concat)
flatten_1 = layers.Flatten()(conv_c)
drop_1 = layers.Dropout(0.2)(flatten_1)
dense_1 = layers.Dense(1)(drop_1)

discriminator = Model(inputs=[fake_image_input, real_image_input], outputs=dense_1)
discriminator.summary()
plot_model(discriminator, to_file="discriminator.png", show_shapes=True)

# Goal of the discriminator is to distinguish real images from fake images
# The Discriminator is given a real image and a fake image, and it outputs a real/fake score

feature_block_dim = (2, 2, 2048)

generator = Sequential(
    [
        layers.Input(shape=feature_block_dim),
        layers.Conv2DTranspose(128, kernel_size=4, strides=4, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2DTranspose(256, kernel_size=4, strides=4, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2DTranspose(512, kernel_size=4, strides=4, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(3, kernel_size=5, padding="same", activation="sigmoid"),
    ]
)
generator.summary()
plot_model(generator, to_file="generator.png", show_shapes=True)


# The Feature Block Generator
from tensorflow.keras.applications import InceptionV3


fb_generator = InceptionV3(include_top=False, weights='imagenet', input_shape=tile_size)





class TileGAN(keras.Model):
    def __init__(self, generator, discriminator, feature_block_generator):
        super(TileGAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.feature_block_generator = feature_block_generator
    
    def compile(self, g_optimizer, d_optimizer, loss_fn):
        super(TileGAN, self).compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.loss_fn = loss_fn
        self.g_loss = keras.metrics.Mean(name='g_loss')
        self.d_loss = keras.metrics.Mean(name='d_loss')
    
    def call(self,inputs):
        return self.generator(self.feature_block_generator(inputs))

    def save(self, folder):
        self.generator.save(os.path.join(folder + 'model_g.h5'))
        self.discriminator.save(os.path.join(folder + 'model_d.h5'))

    @property
    def metrics(self):
        return [self.g_loss, self.d_loss]
    
    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]
        latent_blocks = self.feature_block_generator(real_images)

        # Training Discriminator
        gen_images = self.generator(latent_blocks)
        com_images = tf.concat([real_images, gen_images], axis=0)
        ref_images = tf.concat([real_images, real_images], axis=0)

        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )

        # introduce noise
        labels += 0.05 * tf.random.uniform(tf.shape(labels))

        with tf.GradientTape() as disc_tape:
            disc_output = self.discriminator([com_images, ref_images])
            disc_loss = self.loss_fn(labels, disc_output)
        disc_grads = disc_tape.gradient(disc_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(disc_grads, self.discriminator.trainable_weights))

        gen_labels = tf.zeros((batch_size, 1))

        with tf.GradientTape() as gen_tape:
            disc_output = self.discriminator([self.generator(latent_blocks), real_images])
            gen_loss = self.loss_fn(gen_labels, disc_output)
        gen_grads = gen_tape.gradient(gen_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(gen_grads, self.generator.trainable_weights))

        self.d_loss.update_state(disc_loss)
        self.g_loss.update_state(gen_loss)

        return {
            'g_loss': self.g_loss.result(),
            'd_loss': self.d_loss.result(),
        }


    
gan = TileGAN(generator, discriminator, fb_generator)


from tensorflow import keras
class GANCallBack(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        self.model.save('gan_model')
        if logs['g_loss'] < 0.01:
            self.model.stop_training = True
            print("Training stopped")




gan.compile(
    g_optimizer=keras.optimizers.Adam(1e-4),
    d_optimizer=keras.optimizers.Adam(1e-4),
    loss_fn=keras.losses.BinaryCrossentropy(),
    callbacks=[GANCallBack()]
)