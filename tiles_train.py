from tiles_gan import gan, tile_size
from gen_and_aug import datagen

gan.fit_generator(
    datagen(
        path='../data/train',
        image_size=tile_size[:-1],
    ),
    steps_per_epoch=10,
    epochs=10,
    validation_data=datagen(
        path='../data/test',
        image_size=tile_size[:-1],
    ),
    validation_steps=10,
)

