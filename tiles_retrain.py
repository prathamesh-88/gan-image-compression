from tiles_gan import gan, tile_size, GANCallBack, early_stop
from gen_and_aug import datagen
import os


if __name__ == "__main__":
    image_path = "./images/"
    gan.load("gan_model")
    gan.fit(
        datagen(
            path= image_path,
            image_size=tile_size,
        ),
        # steps_per_epoch=10,
        epochs=200,
        callbacks=[GANCallBack(), early_stop]
    )

    gan.save("gan_model")