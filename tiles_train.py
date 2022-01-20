from tiles_gan import gan, tile_size, GANCallBack
from gen_and_aug import datagen
import os


if __name__ == "__main__":
    image_path = './images/'

    gan.fit(
        datagen(
            path= image_path,
            image_size=tile_size,
        ),
        # steps_per_epoch=10,
        epochs=20,
        callbacks=[GANCallBack()]
    )
    

    
    gan.save("gan_model")

