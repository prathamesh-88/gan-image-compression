from gan import GAN, enc, gen, disc, IMAGE_SIZE
from tensorflow.keras.regularizers import L1
from gen_and_aug import datagen



gan = GAN(enc, gen, disc)
gan.compile({
    "encoder": "adam",
    "generator": "adam",
    "discriminator": "adam",
}, {
    "BCE": "binary_crossentropy",
    "L1": L1,
    "discriminator": "binary_crossentropy",
})

epochs = 10
dataset = datagen("./images", IMAGE_SIZE)
for i in epochs:
    print("Epoch:", i)
    gan.train_epoch(dataset)
    gan.generator.save_weights(f"gan_weights/gan_weights_{i}.h5")
    gan.encoder.save_weights(f"gan_weights/encoder_weights_{i}.h5")
    gan.discriminator.save_weights(f"gan_weights/discriminator_weights_{i}.h5")
    

