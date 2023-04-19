from keras import layers
import keras
from hydra import config
from keras_nlp.layers import TransformerEncoder
import numpy as np
import tensorflow as tf

class BoardEmbedding(layers.Layer):

    # self.board_embedding_old = keras.Sequential([
    #     layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding='same'),
    #     layers.Dropout(0.5),
    #     layers.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding='same'),
    #     layers.Dropout(0.5),
    #     layers.Conv2D(filters=256, kernel_size=(3, 3), activation="relu", padding='same'),
    #     layers.Dropout(0.5),
    #     layers.Flatten(),
    #     layers.Dense(config.embed_dim, activation="relu"),
    #     layers.Reshape((1, -1), name='board_embedding')
    # ])

    def __init__(self):
        super(BoardEmbedding, self).__init__()

        self.board_embedding = keras.Sequential([
            layers.Conv2D(filters=256, kernel_size=(3, 3), activation="relu", padding='same'),
            # layers.Dropout(0.5),
            # layers.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding='same'),
            # layers.Dropout(0.5),
            # layers.Conv2D(filters=256, kernel_size=(3, 3), activation="relu", padding='same'),
            # layers.Dropout(0.5),
            layers.Reshape((64, 256), name='board_embedding'),
        ])
        # --> Custom Transformer Design: 8x8x12 = 768
        self.flatten_layer = layers.Flatten()
        self.board_dense_embedding = layers.Dense(config.embed_dim, activation="relu")
        self.positional_encodings = self.create_positional_encodings(64, 256)

    def create_positional_encodings(self, input_dim, output_dim):
        positional_encodings = tf.range(start=0, limit=input_dim, delta=1, dtype=tf.float32)
        div_term = tf.exp(tf.range(0, output_dim, 2, dtype=tf.float32) * -(tf.math.log(10000.0) / output_dim))
        positional_encodings = tf.expand_dims(positional_encodings, 1)
        positional_encodings *= div_term
        positional_encodings = tf.stack([tf.sin(positional_encodings), tf.cos(positional_encodings)], axis=2)
        positional_encodings = tf.reshape(positional_encodings, [-1, output_dim])
        positional_encodings = tf.expand_dims(positional_encodings, 0)
        return positional_encodings

    def __call__(self, inputs):
        dense_embeddings = self.board_embedding(inputs)
        # position_embeddings = self.positional_encodings
        # combined_embeddings = dense_embeddings + position_embeddings
        return dense_embeddings
        # return self.board_embedding(inputs)

















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

