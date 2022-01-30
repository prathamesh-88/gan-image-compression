import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.applications import InceptionV3

INPUT_SHAPE = (500, 500, 3)

inception = InceptionV3(include_top=False, weights='imagenet', input_shape=INPUT_SHAPE)
# print(inception.summary())
# print(inception.output_shape)
# model = models.Sequential([
#     inception, 
#     ...
# ])

# Plot keras model
from tensorflow.keras.utils import plot_model
plot_model(inception, to_file='model.png')