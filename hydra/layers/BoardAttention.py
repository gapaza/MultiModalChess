from keras import layers
import keras
from hydra import config
import numpy as np
import tensorflow as tf



from hydra.board_attention.ShiftedPatchTokenization import ShiftedPatchTokenization
from hydra.board_attention.PatchEncoder import PatchEncoder
from hydra.board_attention.MultiHeadAttentionLSA import MultiHeadAttentionLSA
from hydra.board_attention.build import mlp, diag_attn_mask


class BoardAttention(layers.Layer):

    def __init__(self):
        super(BoardAttention, self).__init__()

        self.patch_tokenizer = ShiftedPatchTokenization()
        self.patch_encoder = PatchEncoder()

        # --> Visual Transformer
        self.norm_1 = layers.LayerNormalization(epsilon=1e-6)
        self.attn_1 = MultiHeadAttentionLSA(num_heads=config.visual_transformer_heads, key_dim=config.embed_dim, dropout=0.1)
        self.add_1 = layers.Add()
        self.norm_2 = layers.LayerNormalization(epsilon=1e-6)
        self.add_2 = layers.Add()



    def __call__(self, inputs):

        (tokens, _) = self.patch_tokenizer(inputs)
        encoded_patches = self.patch_encoder(tokens)

        # Transformer
        for _ in range(config.visual_transformer_layers):
            x1 = self.norm_1(encoded_patches)
            attention_output = self.attn_1(x1, x1, attention_mask=diag_attn_mask)
            x2 = self.add_1([attention_output, encoded_patches])
            x3 = self.norm_2(x2)
            x3 = mlp(x3, hidden_units=config.visual_transformer_units, dropout_rate=0.1)
            encoded_patches = self.add_2([x3, x2])

        return encoded_patches



