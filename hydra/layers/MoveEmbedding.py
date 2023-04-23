import tensorflow as tf
from keras import layers
import numpy as np
from hydra import config


class MoveEmbedding(layers.Layer):

    def __init__(self):
        super(MoveEmbedding, self).__init__()

        # --> Token Embeddings
        self.token_embeddings = layers.Embedding(
            config.vocab_size, config.embed_dim, name="word_embedding"
        )

        # --> Position Embeddings
        self.token_position_embeddings = layers.Embedding(
            input_dim=config.seq_length,
            output_dim=config.embed_dim,
            weights=[self.get_pos_encoding_matrix(config.seq_length, config.embed_dim)],
            name="position_embedding",
        )

    def __call__(self, inputs):
        token_embeddings = self.token_embeddings(inputs)
        token_position_embeddings = self.token_position_embeddings(tf.range(start=0, limit=config.seq_length, delta=1))
        move_embedding = token_embeddings + token_position_embeddings
        return move_embedding

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









