
from numpy import block
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

INPUT_SHAPE = (64, 64, 27)
OUTPUT_SHAPE = (256, 256, 3)

e_in1 = layers.Input(shape=INPUT_SHAPE)
con1 = layers.Conv2D(64, kernel_size=3, strides=1, padding='same')(e_in1)
le1 = layers.LeakyReLU()(con1)

conT1 = layers.Conv2DTranspose(128, kernel_size=2, strides=2, padding='same')(le1)

def block1(input_tensor):
    bcon1 = layers.Conv2D(128, kernel_size=3, strides=1, padding='same')(input_tensor)
    ble1 = layers.LeakyReLU()(bcon1)
    bcon2 = layers.Conv2D(128, kernel_size=3, strides=1, padding='same')(ble1)
    return bcon2

def block2(input_tensor):
    bcon1 = layers.Conv2D(128, kernel_size=3, strides=1, padding='same')(input_tensor)
    ble1 = layers.LeakyReLU()(bcon1)
    bcon2 = layers.Conv2D(128, kernel_size=3, strides=1, padding='same')(ble1)
    return bcon2

def block3(input_tensor):
    bcon1 = layers.Conv2D(128, kernel_size=3, strides=1, padding='same')(input_tensor)
    ble1 = layers.LeakyReLU()(bcon1)
    bcon2 = layers.Conv2D(128, kernel_size=3, strides=1, padding='same')(ble1)
    return bcon2




dblock1 = layers.Add()([block1(conT1), conT1])
dblock2 = layers.Add()([block2(dblock1), dblock1])
dblock3 = layers.Add()([block3(dblock2), dblock2])

con2 = layers.Conv2D(32, kernel_size=3, strides=1, padding='same')(dblock3)
le2 = layers.LeakyReLU()(con2)
conT2 = layers.Conv2DTranspose(32, kernel_size=2, strides=2, padding='same')(le2)

con3 = layers.Conv2D(16, kernel_size=3, strides=1, padding='same')(conT2)
le3 = layers.LeakyReLU()(con3)

con4 = layers.Conv2D(3, kernel_size=3, strides=1, padding='same')(le3)
out = tf.tanh(con4)
# out = tf.nn.relu(out)


model = keras.Model(inputs=e_in1, outputs=out)

if __name__ == "__main__":
    model.summary()
    print(model.output_shape)
    from tensorflow.keras.utils import plot_model
    # plot_model(model, to_file='diagrams/generator.png', show_shapes=True)