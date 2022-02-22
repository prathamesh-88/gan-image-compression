from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array

batch_size = 8
def datagen(path, image_size):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
    )

    train_generator = train_datagen.flow_from_directory(
        path,
        target_size=image_size[:-1],
        batch_size= batch_size,
        class_mode=None, # since there is no inference data to train against
        shuffle=True,
    )

    return train_generator

if __name__ == '__main__':
    datagen('./image  ')