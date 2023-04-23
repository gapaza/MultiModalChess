from keras import layers
import keras
from hydra import config
import numpy as np
import tensorflow as tf



from hydra.board_attention.ShiftedPatchTokenization import ShiftedPatchTokenization
from hydra.board_attention.PatchEncoder import PatchEncoder
from hydra.board_attention.MultiHeadAttentionLSA import MultiHeadAttentionLSA
from hydra.board_attention.build import mlp, diag_attn_mask


class BoardEmbedding(layers.Layer):

    def __init__(self):
        super(BoardEmbedding, self).__init__()

        self.board_embedding = keras.Sequential([
            layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'),
            layers.Reshape((64, 64)),
        ])


    def __call__(self, inputs):
        outputs = self.board_embedding(inputs)
        return outputs

