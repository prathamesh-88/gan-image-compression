from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array


def preprocess(train_image):
    train_image = (train_image - 127.5) / 127.5
    return train_image

def datagen(path, image_size, batch_size = 4):
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess,
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