
import os, time
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from gen_and_aug import datagen, preprocess, postprocess
from tensorflow.keras.preprocessing.image import array_to_img, load_img, img_to_array


DEBUG = True
BATCH_SIZE = 4
DATA_PATH = os.path.join(".", "images")
IMAGE_SIZE = (256, 256, 3)

image_set = datagen(DATA_PATH, IMAGE_SIZE, BATCH_SIZE)


# Loading Models
# from generator import INPUT_SHAPE, OUTPUT_SHAPE
from discriminator import LATENT_CHANNELS
from tensorflow.keras import Model
def enc() -> Model:
    from encoder import Encoder
    input_shape = (None, *IMAGE_SIZE)
    encoder = Encoder(LATENT_CHANNELS, IMAGE_SIZE)
    encoder.build(input_shape)
    return encoder

def gen() -> Model:
    from generator import model as generator
    return generator

def disc() -> Model:
    from discriminator import model as discriminator
    return discriminator

encoder = enc()
generator = gen()
discriminator = disc()


def introduce_noise(latent_point):
    # return preprocess(tf.cast(postprocess(latent_point), tf.uint8))
    vec_127 = tf.fill(shape=tf.shape(latent_point), value=127.5) # Try the line below if this doesn't work
    # vec_127 = tf.constant(127.5, shape=tf.shape(latent_point), dtype=tf.float32)

    postprocessed = tf.add(
        tf.multiply(
            latent_point, 
            vec_127
        ),
        vec_127
    )
    floored = tf.floor(postprocessed)
    preprocessed = tf.divide(
        tf.subtract(
            floored,
            vec_127
        ),
        vec_127
    )
    return preprocessed

# GAN starts here
# Setting optimizers and losses
# Losses
from tensorflow.keras.losses import BinaryCrossentropy, MeanAbsoluteError
cross_entropy = BinaryCrossentropy(from_logits=True)
mean_absolute_error = MeanAbsoluteError()

def discriminator_loss(real_output, fake_output):
    real_labels = tf.random.uniform(minval=0.855, maxval=0.999,shape=[BATCH_SIZE, 1])
    fake_labels = tf.random.uniform(minval=0.001, maxval=0.145,shape=[BATCH_SIZE, 1])
    real_loss   = cross_entropy(real_labels, real_output)
    fake_loss   = cross_entropy(fake_labels, fake_output)
    return real_loss + fake_loss

def generator_loss(fake_output, real_image, fake_image):
    labels = tf.random.uniform(minval=0.855, maxval=0.999,shape=[BATCH_SIZE, 1])
    loss = cross_entropy(labels, fake_output)
    rg = 2 * mean_absolute_error(real_image, fake_image)
    return loss+rg

# Optimizers
from tensorflow.keras.optimizers import Adam as Opt
generator_optimizer = Opt(1e-4)
discriminator_optimizer = Opt(1e-4)
encoder_optimizer = Opt(1e-4)


# Checkpoints Mechanism
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


# Training Functions
@tf.function
def train_step(real_images):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape, tf.GradientTape() as enc_tape:

        real_encoded = introduce_noise(encoder(real_images, training= True))
        generated_images = generator(real_encoded, training= True)

        real_in = [real_encoded, real_images]
        fake_in = [real_encoded, generated_images]

        real_out = discriminator(real_in, training= True)
        fake_out = discriminator(fake_in, training= True)

        disc_loss = discriminator_loss(real_out, fake_out)
        gen_loss = generator_loss(fake_out, real_images, generated_images)

    disc_grads = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(disc_grads, discriminator.trainable_variables))

    gen_grads = gen_tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gen_grads, generator.trainable_variables))

    enc_grads = enc_tape.gradient(gen_loss, encoder.trainable_variables)
    encoder_optimizer.apply_gradients(zip(enc_grads, encoder.trainable_variables))

    return (gen_loss, disc_loss)


def generate_and_save_images(generator, encoder, epoch, test_input):
    encoded = encoder(test_input)
    predictions = generator(encoded)
    #strip
    count = predictions.shape[0]
    for i in range(count):
        plt.subplot(1, count, i + 1)
        img = array_to_img((predictions[i] * 127.5 + 127.5) / 255)
        plt.imshow(img)
        plt.axis('off')

    plt.savefig('results/strips/epoch_strip_{:04d}.png'.format(epoch))
    plt.clf()

    compare = np.hstack([test_input[0], predictions[0]])
    compare = compare * 127.5 + 127.5
    array_to_img(compare).save(f"results/image_compare/{epoch:04d}_compare_image.png")

    dir_ = os.path.join(".", "images", "train")
    image = os.path.join(dir_, os.listdir(dir_)[0])
    image = load_img(image)
    # image = tf.image.resize(image, (256, 256))
    ti = img_to_array(image) / 127.5 - 1
    pr = generator(encoder(tf.expand_dims(ti,axis=0), training=False), training=False)[0]
    compare = np.hstack([ti, pr])


    compare = compare * 127.5 + 127.5
    array_to_img(compare).save(f"results/same_img/{epoch:04d}_gradual_compare.png")



dir_ = os.path.join(".", "images", "train")
steps_per_epoch = len(os.listdir(dir_)) // BATCH_SIZE

def train(epochs):
    dataset = image_set
    checkpoint.restore(tf.train.latest_checkpoint(CHECKPOINT_DIR)).expect_partial()

    if manager.latest_checkpoint:
        print(f"INFO: {manager.latest_checkpoint} Restored")
    else:
        print("INFO: Starting from scratch")
    
    print("DEBUG: Training Started!") if DEBUG else None

    for epoch in tqdm(range(epochs), desc='Epoch'):
        gen_losses = []
        disc_losses = []
        start = time.perf_counter()
        for i, image_batch in enumerate(tqdm(dataset, desc=f'Train Steps of Epoch {epoch + 1}')):
            gl, dl = train_step(image_batch)
            gen_losses.append(gl)
            disc_losses.append(dl)
            if i >= steps_per_epoch:
                break
        dataset.on_epoch_end()

        generate_and_save_images(generator, encoder, epoch + 1, dataset[0])

        if (epoch + 1) % 3 == 0:
            checkpoint.save(file_prefix=CHECKPOINT_PREFIX)
        
        gl = sum(gen_losses) / len(gen_losses)
        dl = sum(disc_losses) / len(disc_losses)
        print(f"\nEpoch {epoch + 1}: [Finished in {(time.perf_counter() - start):.2f} sec] \nMetrics: Avg. GLoss: {gl:.2f}, Avg. DLoss: {dl:.2f}\n")

    checkpoint.save(file_prefix=CHECKPOINT_PREFIX)
    




