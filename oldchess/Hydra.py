from keras import Model
from keras import layers
import keras
import pickle
import numpy as np
import tensorflow as tf
import os
from keras_nlp.layers import TransformerEncoder, TokenAndPositionEmbedding, MaskedLMHead
from TokenizerManager import TokenizerManager






class Hydra(layers.Layer):

    def __init__(self, vocab_size, seq_length=128, embed_dim=256):
        super(Hydra, self).__init__()
        self.vocab_size = vocab_size
        self.seq_length = seq_length


        # --> Board Embedding
        self.board_conv2d_1 = layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu")
        self.board_conv2d_1_dropout = layers.Dropout(0.5)
        self.board_conv2d_2 = layers.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding='same')
        self.board_conv2d_2_dropout = layers.Dropout(0.5)
        self.board_flatten = layers.Flatten()
        self.board_dense = layers.Dense(embed_dim, activation="relu")
        self.board_reshape = layers.Reshape((1, -1), name='board_embedding')

        # --> Move Embedding
        self.word_embeddings = layers.Embedding(
            self.vocab_size, embed_dim, name="word_embedding"
        )
        self.pos_embeddings = layers.Embedding(
            input_dim=seq_length,
            output_dim=embed_dim,
            weights=[self.get_pos_encoding_matrix(seq_length, embed_dim)],
            name="position_embedding",
        )
        # self.move_embedding = TokenAndPositionEmbedding(
        #     self.vocab_size,
        #     self.seq_length,
        #     embedding_dim=embed_dim,
        #     weights=[self.get_pos_encoding_matrix(seq_length, embed_dim)],
        #     name='move_embedding',
        # )






        # --> Modality Fusion
        self.modality_fusion = layers.Concatenate(axis=1, name='modality_fusion')

        # --> Encoder Stack
        # sequential layer
        self.encoder_stack = keras.Sequential([
            TransformerEncoder(2048, 12),
            layers.Dropout(0.5),
            TransformerEncoder(2048, 12),
            layers.Dropout(0.5),
            TransformerEncoder(2048, 12),
            layers.Dropout(0.5),
            TransformerEncoder(2048, 12),
            # TransformerEncoder(2048, 12),
            # TransformerEncoder(2048, 12)
        ])

        # ------------------
        # -- Output Heads --
        # ------------------

        # --> Next Move Prediction
        self.next_move_avg = layers.GlobalAveragePooling1D()
        self.next_move_dropout = layers.Dropout(0.5)
        self.next_move_prediction = layers.Dense(self.vocab_size, activation="softmax", name='next_move_prediction')

        # --> Board Mask Prediction
        self.board_mask_dense = layers.Dense(8 * 8 * 12, activation="linear")
        self.board_prediction = layers.Reshape((8, 8, 12), name='board_prediction')

        # --> Move Mask Prediction
        # self.masked_lm = MaskedLMHead(
        #     embedding_weights=self.move_embedding.embeddings,
        #     activation='softmax',
        #     name='move_mask_prediction',
        # )

    def __call__(self, board_inputs, move_inputs, mask=None):

        # --> Board Embedding
        board_embedding = self.board_conv2d_1(board_inputs)
        board_embedding = self.board_conv2d_1_dropout(board_embedding)
        board_embedding = self.board_conv2d_2(board_embedding)
        board_embedding = self.board_conv2d_2_dropout(board_embedding)
        board_embedding = self.board_flatten(board_embedding)
        board_embedding = self.board_dense(board_embedding)
        board_embedding = self.board_reshape(board_embedding)

        # --> Move Embedding: (None, None, 256)
        # move_embedding = self.move_embedding(move_inputs)
        word_embedding = self.word_embeddings(move_inputs)
        position_embedding = self.pos_embeddings(tf.range(start=0, limit=self.seq_length, delta=1))
        move_embedding = word_embedding + position_embedding


        # --> Combine Board and Move Embeddings
        encoder_inputs = self.modality_fusion([board_embedding, move_embedding])

        # --> Encoder Stack
        encoder_outputs = self.encoder_stack(encoder_inputs)

        # --> Output Heads
        # outputs = []
        # for head in self.heads:
        #     if head == 'next_move':
        #         next_move = self.next_move_avg(encoder_outputs)
        #         next_move = self.next_move_dropout(next_move)
        #         next_move = self.next_move_prediction(next_move)
        #         outputs.append(next_move)
        #     elif head == 'board_mask':
        #         encoded_board_input = encoder_outputs[:, 0, :]
        #         board_mask = self.board_mask_dense(encoded_board_input)
        #         board_mask = self.board_prediction(board_mask)
        #         outputs.append(board_mask)
        #     elif head == 'move_mask':
        #         encoded_move_input = encoder_outputs[:, 1:, :]
        #         move_mask = self.masked_lm(encoded_move_input)
        #         outputs.append(move_mask)
        #     else:
        #         raise Exception('Invalid Head: ', head)
        return encoder_outputs

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


if __name__ == '__main__':
    hydra = Hydra()


