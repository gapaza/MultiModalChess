from keras import layers
from hydra import config
from keras_nlp.layers import TransformerEncoder
import keras

class Encoder(layers.Layer):

    def __init__(self):
        super(Encoder, self).__init__()

        # --> Encoders
        self.encoder_stack = keras.Sequential([
            TransformerEncoder(config.encoder_dense_dim, config.encoder_heads),
            TransformerEncoder(config.encoder_dense_dim, config.encoder_heads),
            TransformerEncoder(config.encoder_dense_dim, config.encoder_heads),
            TransformerEncoder(config.encoder_dense_dim, config.encoder_heads),
            TransformerEncoder(config.encoder_dense_dim, config.encoder_heads),
            TransformerEncoder(config.encoder_dense_dim, config.encoder_heads),
            # TransformerEncoder(config.encoder_dense_dim, config.encoder_heads),
            # TransformerEncoder(config.encoder_dense_dim, config.encoder_heads),
            # TransformerEncoder(config.encoder_dense_dim, config.encoder_heads),
            # TransformerEncoder(config.encoder_dense_dim, config.encoder_heads),
            # TransformerEncoder(config.encoder_dense_dim, config.encoder_heads),
            # TransformerEncoder(config.encoder_dense_dim, config.encoder_heads),
            # TransformerEncoder(config.encoder_dense_dim, config.encoder_heads),
            # TransformerEncoder(config.encoder_dense_dim, config.encoder_heads),
        ])




    def __call__(self, inputs):
        encoder_output = self.encoder_stack(inputs)
        return encoder_output