import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(gpus[0], True)
tf.config.run_functions_eagerly(True)

from gan import GAN, enc, gen, disc, IMAGE_SIZE
from tensorflow.keras.regularizers import L1
from gen_and_aug import datagen
from tensorflow.keras.losses import binary_crossentropy, MAE
from tensorflow.keras.optimizers import Adam

gan = GAN(enc, gen, disc)
adm = Adam(beta_1=0.5)
gan.compile({
    "encoder": adm,
    "generator": adm,
    "discriminator": adm,
}, {
    "BCE": binary_crossentropy,
    "L1": MAE,
    "discriminator": binary_crossentropy,
})

epochs = 20
dataset = datagen("./images", IMAGE_SIZE)
for i in range(epochs):
    print("Epoch:", i)
    gan.train_epoch(dataset, i)
    gan.generator.save_weights(f"gan_weights/gan_weights_{i}.h5")
    gan.encoder.save_weights(f"gan_weights/encoder_weights_{i}.h5")
    gan.discriminator.save_weights(f"gan_weights/discriminator_weights_{i}.h5")
    
    

