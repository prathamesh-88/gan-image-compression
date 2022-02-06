
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


class Encoder(keras.Model):
    
    def __init__(self, encoder_channels=27, input_shape=(256,256,3)):
        super(Encoder, self).__init__()
        
        self.dim = input_shape;
        
        # Input: 256x256x3 / Output: 128x128x64
        self.conv1 = Sequential(
            [
                layers.Conv2D(64, kernel_size=5, strides=2, padding='same'),
                layers.LeakyReLU()
            ], name='conv1'
        )

        # Input: 128x128x64 / Output: 64x64x128
        self.conv2 = Sequential(
            [
                layers.Conv2D(128, kernel_size=5, strides=2, padding='same'),
                layers.LeakyReLU()
            ], name='conv2'
        
        )
        
        # Input: 128x128x64 / Output: 64x64x128
        self.block1 = Sequential(
            [
                layers.Conv2D(128, kernel_size=3, strides=1, padding='same'),
                layers.LeakyReLU(),
                layers.Conv2D(128, kernel_size=3, strides=1, padding='same'),
            ], name="block1"
        )

        # Input: 128x128x64 / Output: 64x64x128
        self.block2 = Sequential(        
            [
                layers.Conv2D(128, kernel_size=3, strides=1, padding='same'),
                layers.LeakyReLU(),
                layers.Conv2D(128, kernel_size=3, strides=1, padding='same')
            ], name="block2"
        )

        # Input: 128x128x64 / Output: 64x64x128
        self.block3 = Sequential(
            [
                layers.Conv2D(128, kernel_size=3, strides=1, padding='same'),
                layers.LeakyReLU(),
                layers.Conv2D(128, kernel_size=3, strides=1, padding='same')
            ], name="block3"
        )

        
        # Input: 128x128x64 / Output: 64x64x27
        self.conv3 = Sequential(
            [
                layers.Conv2D(encoder_channels, kernel_size=5, strides=1 ,activation=keras.activations.tanh, padding='same'),       
            ], name='conv3'
        )
        
        
        
        # print(self.conv1.summary())
        # print(self.conv2.summary())
        # print(self.block1.summary())
        # print(self.block2.summary())
        # print(self.block3.summary())        
        # print(self.conv3.summary())
    
    
    def call(self, inputs):
        layer_1 = self.conv1(inputs)
        layer_2 = self.conv2(layer_1)
        layer_3 = layers.Add()([self.block1(layer_2), layer_2])
        layer_4 = layers.Add()([self.block2(layer_3), layer_3])
        layer_5 = layers.Add()([self.block3(layer_4), layer_4])
        layer_6 = self.conv3(layer_5)
        
        
        return layer_6
    
    def build_graph(self):
        x = layers.Input(shape=(self.dim))
        return keras.Model(inputs=[x], outputs=self.call(x))
        
        
if __name__=='__main__':
    input_shape = (None, 256, 256,3)
    encoder = Encoder(input_shape=input_shape[1:])
    encoder.build(input_shape)
    # tf.keras.utils.plot_model(
    #     encoder.build_graph(),                      # here is the trick (for now)
    #     to_file='./diagrams/encoder.png', dpi=96,              # saving  
    #     show_shapes=True, show_layer_names=True,  # show shapes and layer name
    #     expand_nested=False                       # will show nested block
    # )
    encoder.summary()


    
    
    