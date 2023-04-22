from keras import layers
import keras
from hydra import config
import tensorflow as tf

from hydra.board_attention.ShiftedPatchTokenization import ShiftedPatchTokenization
from hydra.board_attention.PatchEncoder import PatchEncoder
from hydra.board_attention.MultiHeadAttentionLSA import MultiHeadAttentionLSA
from hydra.board_attention.build import mlp, diag_attn_mask


class BoardEmbedding(layers.Layer):

    def __init__(self):
        super(BoardEmbedding, self).__init__()
        self.board_embedding = keras.Sequential([

        ])


    def __call__(self, inputs):
        output = self.board_embedding(inputs)
        return output