import tensorflow as tf
import numpy as np
from tensorflow.keras import Model, Sequential, layers
import os

IMAGE_SIZE = (256, 256, 3)
LATENT_CHANNELS = 27

# Assuming Latent shape to be (64, 64, LATENT_CHANNELS)

# ---------------- Callback to save weights ----------------
"""
We should use checkpoints to save weights
`from tensorflow.keras.callbacks import ModelCheckpoint`
"""


# ---------------- Discriminator Model ----------------
lv = layers.Input(shape= (32, 32, LATENT_CHANNELS))
ll1 = layers.Conv2DTranspose(12, (3, 3), strides=2, padding='same')(lv)
ll1_lr = layers.LeakyReLU(alpha=.2)(ll1)
ll2 = layers.Conv2DTranspose(16, (3, 3), strides=2, padding='same')(ll1_lr)
ll2_lr = layers.LeakyReLU(alpha=.2)(ll2)
ll3 = layers.Conv2DTranspose(24, (3, 3), strides=2, padding='same')(ll2_lr)
ll3_lr = layers.LeakyReLU(alpha=.2)(ll3)
ll4 = layers.Conv2DTranspose(3, (3, 3), strides=1, padding='same')(ll3_lr)
ll4_lr = layers.LeakyReLU(alpha=.2)(ll4)

ri = layers.Input(shape=IMAGE_SIZE)

cc = layers.Concatenate(-1)([ri, ll4_lr])
cv1 = layers.Conv2D(32, kernel_size=3, strides=1)(cc)
cv1_lr = layers.LeakyReLU(alpha=.2)(cv1)
cv1_lr_d = layers.Dropout(.3)(cv1_lr)

cv2 = layers.Conv2D(64, kernel_size=5, strides=2, padding='same')(cv1_lr_d)
cv2_lr = layers.LeakyReLU(alpha=.2)(cv2)
cv2_lr_d = layers.Dropout(.3)(cv2_lr)

cv3 = layers.Conv2D(32, kernel_size=3, strides=2, padding='same')(cv2_lr_d)
cv3_lr = layers.LeakyReLU(alpha=.2)(cv3)
cv3_lr_d = layers.Dropout(.3)(cv3_lr)

cv4 = layers.Conv2D(16, kernel_size=3, strides=2)(cv3_lr_d)
cv4_lr = layers.LeakyReLU(alpha=.2)(cv4)
cv4_lr_d = layers.Dropout(.3)(cv4_lr)

cv5 = layers.Conv2D(8, kernel_size=3, strides=2)(cv4_lr_d)
cv5_lr = layers.LeakyReLU(alpha=.2)(cv5)
cv5_lr_d = layers.Dropout(.3)(cv5_lr)

cv_op = tf.nn.tanh(cv5_lr_d)
fl = layers.Flatten()(cv_op)
dd1 = layers.Dense(1000, activation='sigmoid')(fl)
dd2 = layers.Dense(100, activation='sigmoid')(dd1)
dd3 = layers.Dense(1, activation='sigmoid')(dd2)


model = Model(inputs=[lv, ri], outputs=dd3)

if __name__ == "__main__":
    print(model.summary())
    # plot model
    tf.keras.utils.plot_model(model, to_file='diagrams/discriminator.png', show_shapes=True)