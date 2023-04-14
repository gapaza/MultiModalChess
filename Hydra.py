from keras import Model
from keras import layers
import pickle
import os
from keras_nlp.layers import TransformerEncoder, TokenAndPositionEmbedding, MaskedLMHead
from TokenizerManager import TokenizerManager

class Hydra(Model):

    def __init__(self, mode='pretraining'):
        super(Hydra, self).__init__()
        self.mode = mode


        # --> Tokenizer
        self.tokenizer_manager = TokenizerManager()
        self.tokenizer = self.tokenizer_manager.moves_tokenizer

        # --> Board Embedding
        self.board_conv2d_1 = layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu")
        self.board_conv2d_2 = layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding='same')
        self.board_flatten = layers.Flatten()
        self.board_dense = layers.Dense(256, activation="relu")
        self.board_reshape = layers.Reshape((1, -1), name='board_embedding')

        # --> Move Embedding
        self.max_move_length = 100
        self.move_embedding = TokenAndPositionEmbedding(self.token_count, self.max_move_length, 256,
                                                        name='move_embedding')

        # --> Modality Fusion
        self.modality_fusion = layers.Concatenate(axis=1, name='modality_fusion')

        # --> Encoder Stack
        self.encoder_1 = TransformerEncoder(2048, 12)

        # ------------------
        # -- Output Heads --
        # ------------------

        # --> Next Move Prediction
        self.next_move_avg = layers.GlobalAveragePooling1D()
        self.next_move_dropout = layers.Dropout(0.5)
        self.next_move_prediction = layers.Dense(self.token_count, activation="softmax", name='next_move_prediction')

        # --> Board Mask Prediction
        self.board_mask_dense = layers.Dense(8 * 8 * 12, activation="linear")
        self.board_prediction = layers.Reshape((8, 8, 12), name='board_prediction')

        # --> Move Mask Prediction
        self.masked_lm = MaskedLMHead(
            embedding_weights=self.move_embedding.embeddings,
            activation='softmax',
            name='move_mask_prediction',
        )

        def __call__(self, board_inputs, move_inputs, mask=None):

            # --> Board Embedding
            board_embedding = self.board_conv2d_1(board_inputs)
            board_embedding = self.board_conv2d_2(board_embedding)
            board_embedding = self.board_flatten(board_embedding)
            board_embedding = self.board_dense(board_embedding)
            board_embedding = self.board_reshape(board_embedding)

            # --> Move Embedding: (None, None, 256)
            move_embedding = self.move_embedding(move_inputs)

            # --> Combine Board and Move Embeddings
            encoder_inputs = self.modality_fusion([board_embedding, move_embedding])

            # --> Encoder Stack
            encoder_outputs = self.encoder_1(encoder_inputs, mask=mask)

            # --> Output Heads
            outputs = []
            for head in self.heads:
                if head == 'next_move':
                    next_move = self.next_move_avg(encoder_outputs)
                    next_move = self.next_move_dropout(next_move)
                    next_move = self.next_move_prediction(next_move)
                    outputs.append(next_move)
                elif head == 'board_mask':
                    encoded_board_input = encoder_outputs[:, 0, :]
                    board_mask = self.board_mask_dense(encoded_board_input)
                    board_mask = self.board_prediction(board_mask)
                    outputs.append(board_mask)
                elif head == 'move_mask':
                    encoded_move_input = encoder_outputs[:, 1:, :]
                    move_mask = self.masked_lm(encoded_move_input)
                    outputs.append(move_mask)
                else:
                    raise Exception('Invalid Head: ', head)
            return outputs



if __name__ == '__main__':
    hydra = Hydra()


