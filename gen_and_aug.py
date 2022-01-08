from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array


def datagen(path, image_size):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
    )

    train_generator = train_datagen.flow_from_directory(
        path,
        target_size=image_size[:-1],
        batch_size=32,
        class_mode=None, # since there is no inference data to train against
        shuffle=True,
    )

    return train_generator

