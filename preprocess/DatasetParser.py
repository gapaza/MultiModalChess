import tensorflow as tf
from keras.layers import TextVectorization
import pandas as pd
import numpy as np
import os
import pickle
import random
import re

from hydra import config



class DatasetParser:

    def __init__(self):
        self.vocab = config.vocab
        self.vocab_size = config.vocab_size

        # --> 1. Get Position Data
        print('--> 1. Get Position Data')
        self.all_positions = self.parse_positions_file(config.human_positions_file)
        self.num_positions = config.human_positions_file.split('-')[-1].split('.')[0]

        # --> 2. Create Vectorization Layer
        print('--> 2. Create Vectorization Layer')
        self.vectorize_layer = self.get_vectorize_layer(
            self.all_positions.moves.values.tolist(),
            self.vocab_size,
            config.seq_length,
            special_tokens=["[mask]"],
        )
        self.mask_token_id = self.vectorize_layer(["[mask]"]).numpy()[0][0]

        # --> 3. Encode Moves and Extract Boards
        print('--> 3. Encode Moves and Extract Boards')
        self.all_moves = self.encode(self.all_positions.moves.values.tolist())
        self.all_boards = self.all_positions.board.values.tolist()

        # --> 4. Split into Train and Validation
        print('--> 4. Split into Train and Validation')
        split_idx = int(len(self.all_moves) * 0.8)
        self.train_moves = self.all_moves[:split_idx]
        self.train_boards = self.all_boards[:split_idx]
        self.validation_moves = self.all_moves[split_idx:]
        self.validation_boards = self.all_boards[split_idx:]

        # --> 5. Preprocess Train and Validation Datasets
        print('--> 5. Preprocess Train and Validation Datasets')
        train_x, train_y, train_sample_weights = self.get_masked_input_and_labels(self.train_moves)
        buffer_size = len(train_x)
        self.train_dataset = tf.data.Dataset.from_tensor_slices(
            (train_x, train_y, train_sample_weights, self.train_boards)
        )
        self.train_dataset = self.train_dataset.shuffle(buffer_size).batch(config.batch_size)

        validation_x, validation_y, validation_sample_weights = self.get_masked_input_and_labels(self.validation_moves)
        buffer_size = len(validation_x)
        self.val_dataset = tf.data.Dataset.from_tensor_slices(
            (validation_x, validation_y, validation_sample_weights, self.validation_boards)
        )
        self.val_dataset = self.val_dataset.shuffle(buffer_size).batch(config.batch_size)

        # --> 6. Save Datasets
        print('--> 6. Save Datasets')
        self.train_dataset.save(os.path.join(config.datasets_dir, 'train-dataset-' + str(self.num_positions)))
        self.val_dataset.save(os.path.join(config.datasets_dir, 'val-dataset-' + str(self.num_positions)))


    def parse_positions_file(self, positions_file):
        print('Positions file: ', config.human_positions_file)
        positions = []
        with open(positions_file, 'rb') as f:
            positions = pickle.load(f)
        all_data = pd.DataFrame(positions)

        # --> Conditions for dropping positions
        # 1. No moves exist
        all_data = all_data[all_data['moves'] != '']
        all_data.reset_index(drop=True, inplace=True)
        return all_data

    def get_vectorize_layer(self, texts, vocab_size, max_seq, special_tokens=["[mask]"]):

        # --> Standardazation
        def custom_standardization(input_data):
            lowercase = tf.strings.lower(input_data)
            stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")
            return tf.strings.regex_replace(
                stripped_html, "[%s]" % re.escape("!#$%&'()*+,-./:;<=>?@\^_`{|}~"), ""
            )

        # --> Create Vectorization Layer and Adapt
        vectorize_layer = TextVectorization(
            max_tokens=vocab_size,
            output_mode="int",
            standardize=custom_standardization,
            output_sequence_length=max_seq,
        )
        vectorize_layer.adapt(texts)

        # Insert mask token in vocabulary
        vocab = vectorize_layer.get_vocabulary()
        vocab = vocab[2: vocab_size - len(special_tokens)] + ["[mask]"]
        vectorize_layer.set_vocabulary(vocab)
        return vectorize_layer

    def encode(self, texts):
        encoded_texts = self.vectorize_layer(texts)
        return encoded_texts.numpy()

    def get_masked_input_and_labels(self, inputs):

        # encoded_texts: shape(N, 128) where 128 is the max sequence length
        # - filled with tokenized values
        encoded_texts = inputs


        # inp_mask: shape(N, 128)
        # - filled with True or False denoting whether to mask that token
        # - do not mask special tokens
        inp_mask = np.random.rand(*encoded_texts.shape) < 0.15
        inp_mask[encoded_texts <= 2] = False


        # labels: shape(N, 128)
        # - defines the labels for the masked tokens
        # - all unmasked token positions are set to -1 (ignore)
        labels = -1 * np.ones(encoded_texts.shape, dtype=int)
        labels[inp_mask] = encoded_texts[inp_mask]


        # encoded_texts_masked: shape(N, 128)
        # - modified copy of encoded_texts
        # - masked positions are replaced with mask token id
        encoded_texts_masked = np.copy(encoded_texts)

        # - 90% of masked tokens are predicted
        predict_prob = 1.0
        inp_mask_2mask = inp_mask & (np.random.rand(*encoded_texts.shape) < predict_prob)
        encoded_texts_masked[inp_mask_2mask] = self.mask_token_id  # mask token is the last in the dict

        # - 10% of masked tokens are replaced with random token
        random_prob = 0.0
        inp_mask_2random = inp_mask_2mask & (np.random.rand(*encoded_texts.shape) < random_prob)
        encoded_texts_masked[inp_mask_2random] = np.random.randint(3, self.mask_token_id, inp_mask_2random.sum())

        # sample_weights: shape(N, 128)
        # - defines the weights for the loss function
        # - weights are 0 for all unmasked token positions and 1 for masked token positions
        sample_weights = np.ones(labels.shape)
        sample_weights[labels == -1] = 0

        # y_labels: shape(N, 128)
        # - defines the labels for the masked tokens
        # - is essentially the input to this function
        y_labels = np.copy(encoded_texts)

        return encoded_texts_masked, y_labels, sample_weights


if __name__ == '__main__':
    dp = DatasetParser()