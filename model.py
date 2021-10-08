import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.applications import InceptionV3

INPUT_SHAPE = (500, 500, 3)

inception = InceptionV3(include_top=False, weights='imagenet', input_shape=INPUT_SHAPE)


model = models.Sequential([
    inception, 
    ...
])