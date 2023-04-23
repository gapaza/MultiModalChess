from keras import layers
import keras
from hydra import config
import numpy as np
import tensorflow as tf



from hydra.board_attention.ShiftedPatchTokenization import ShiftedPatchTokenization
from hydra.board_attention.PatchEncoder import PatchEncoder
from hydra.board_attention.MultiHeadAttentionLSA import MultiHeadAttentionLSA


class BoardAttention(layers.Layer):

    def __init__(self):
        super(BoardAttention, self).__init__()

        # --> Patch Embedding
        self.patch_tokenizer = ShiftedPatchTokenization()
        self.patch_encoder = PatchEncoder()

        # --> Visual Transformer
        self.attn_lsa = MultiHeadAttentionLSA(num_heads=config.visual_transformer_heads, key_dim=config.embed_dim, dropout=0.1)
        self.norm_1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm_2 = layers.LayerNormalization(epsilon=1e-6)
        self.add_1 = layers.Add()
        self.add_2 = layers.Add()
        self.dense_1 = layers.Dense(units=config.visual_transformer_dense_dim * 2, activation=tf.nn.gelu)
        self.dense_2 = layers.Dense(units=config.visual_transformer_dense_dim, activation=tf.nn.gelu)
        self.dropout_1 = layers.Dropout(0.1)
        self.dropout_2 = layers.Dropout(0.1)



    def __call__(self, inputs):

        # Embed Patches
        tokens = self.patch_tokenizer(inputs)

        # Embed Patch Positions
        encoded_patches = self.patch_encoder(tokens)

        # Transformer
        for _ in range(config.visual_transformer_layers):
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



