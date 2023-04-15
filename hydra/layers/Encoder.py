from keras import layers
from hydra import config
from keras_nlp.layers import TransformerEncoder
import keras

class Encoder(layers.Layer):

    def __init__(self):
        super(Encoder, self).__init__()

        # --> Encoders
        self.encoder_1 = TransformerEncoder(config.encoder_dense_dim, config.encoder_heads)
        self.encoder_1_dropout = layers.Dropout(0.5)
        self.encoder_2 = TransformerEncoder(config.encoder_dense_dim, config.encoder_heads)
        self.encoder_2_dropout = layers.Dropout(0.5)
        self.encoder_3 = TransformerEncoder(config.encoder_dense_dim, config.encoder_heads)
        self.encoder_3_dropout = layers.Dropout(0.5)



    def __call__(self, inputs):
        encoder_output = self.encoder_1(inputs)
        encoder_output = self.encoder_1_dropout(encoder_output)
        encoder_output = self.encoder_2(encoder_output)
        encoder_output = self.encoder_2_dropout(encoder_output)
        encoder_output = self.encoder_3(encoder_output)
        encoder_output = self.encoder_3_dropout(encoder_output)
        return encoder_output