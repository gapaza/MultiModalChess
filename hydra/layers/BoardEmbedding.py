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

















    def get_pos_encoding_matrix(self, max_len, d_emb):
        pos_enc = np.array(
            [
                [pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)]
                if pos != 0
                else np.zeros(d_emb)
                for pos in range(max_len)
            ]
        )
        pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])  # dim 2i
        pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])  # dim 2i+1
        return pos_enc






    def generate_2d_positional_encoding(self, h, w, d_model):
        h_angles, w_angles = self.get_2d_angles(h, w, d_model)

        pos_encoding = np.zeros((h, w, d_model))

        pos_encoding[:, :, 0::2] = np.sin(h_angles)[:, np.newaxis, :]
        pos_encoding[:, :, 1::2] = np.cos(w_angles)[np.newaxis, :, :]

        pos_encoding = tf.convert_to_tensor(pos_encoding[np.newaxis, ...], dtype=tf.float32)
        return pos_encoding


    def get_2d_angles(self, h, w, d_model):
        h_positions = np.arange(h)[:, np.newaxis]
        w_positions = np.arange(w)[:, np.newaxis]
        div_terms = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

        h_angles = h_positions * div_terms
        w_angles = w_positions * div_terms

        return h_angles, w_angles

