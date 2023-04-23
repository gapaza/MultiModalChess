from keras import layers
import keras
from hydra import config
import numpy as np
import tensorflow as tf

from hydra.board_attention.MultiHeadAttentionLSA import MultiHeadAttentionLSA



class VisualEncoder(layers.Layer):

    def __init__(self):
        super(VisualEncoder, self).__init__()

        # Local Self-Attention
        self.attn_lsa = MultiHeadAttentionLSA(num_heads=config.vt_heads, key_dim=config.embed_dim, dropout=0.1)

        # Other Components
        self.norm_1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm_2 = layers.LayerNormalization(epsilon=1e-6)
        self.add_1 = layers.Add()
        self.add_2 = layers.Add()
        self.dense_1 = layers.Dense(units=config.embed_dim * 2, activation=tf.nn.gelu)
        self.dense_2 = layers.Dense(units=config.embed_dim, activation=tf.nn.gelu)
        self.dropout_1 = layers.Dropout(0.1)
        self.dropout_2 = layers.Dropout(0.1)

    def __call__(self, inputs):
        encoded_patches = inputs
        for _ in range(config.vt_layers):
            x1 = self.norm_1(encoded_patches)

            attention_output = self.attn_lsa(x1, x1)
            x2 = self.add_1([attention_output, encoded_patches])
            x3 = self.norm_2(x2)

            x3 = self.dense_1(x3)
            x3 = self.dropout_1(x3)

            x3 = self.dense_2(x3)
            x3 = self.dropout_2(x3)

            encoded_patches = self.add_2([x3, x2])

        return encoded_patches









