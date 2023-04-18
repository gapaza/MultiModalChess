import tensorflow as tf
from keras.layers import TextVectorization
import pandas as pd
import numpy as np
import os
import pickle
import re
import shutil

from hydra import config
import sys

class DatasetParser:

    def __init__(self):
        self.vocab = config.vocab
        self.vocab_size = config.vocab_size

        # --> 1. Get Position Data
        print('--> 1. Get Position Data')
        self.all_positions = self.load_positions(config.positions_file)
        self.all_positions = self.all_positions.sample(frac=1).reset_index(drop=True)
        self.num_positions = config.positions_file.split('-')[-1].split('.')[0]

        # --> 2. Encode Moves and Extract Boards
        print('--> 2. Encode Moves and Extract Boards')
        self.all_moves = config.encode(self.all_positions.moves.values.tolist())
        self.all_next_moves = config.encode(self.all_positions.next_move.values.tolist())
        self.all_boards = self.all_positions.board.values.tolist()

        # --> 3. Split into Train and Validation
        print('--> 3. Split into Train and Validation')
        split_idx = int(len(self.all_moves) * 0.9)
        self.train_moves = self.all_moves[:split_idx]
        self.train_boards = self.all_boards[:split_idx]
        self.validation_moves = self.all_moves[split_idx:]
        self.validation_boards = self.all_boards[split_idx:]

        # --> 4. Preprocess Train and Validation Datasets
        print('--> 4. Preprocess Train and Validation Datasets')
        # self.train_dataset = self.pretraining_preprocessing(self.train_moves, self.train_boards)
        # self.val_dataset = self.pretraining_preprocessing(self.validation_moves, self.validation_boards)
        self.train_dataset = self.pretraining_sequence_preprocessing(self.train_moves, self.train_boards)
        self.val_dataset = self.pretraining_sequence_preprocessing(self.validation_moves, self.validation_boards)


        # --> 5. Save Datasets
        print('--> 5. Save Datasets')
        train_path = os.path.join(config.datasets_dir, 'train-dataset-' + str(self.num_positions))
        val_path = os.path.join(config.datasets_dir, 'val-dataset-' + str(self.num_positions))
        self.train_dataset.save(train_path)
        self.val_dataset.save(val_path)


    def load_positions(self, positions_file):
        print('Positions file: ', config.positions_file)
        positions = []
        with open(positions_file, 'rb') as f:
            positions = pickle.load(f)
        all_data = pd.DataFrame(positions)

        # --> Conditions for dropping positions
        # 1. No moves exist
        all_data = all_data[all_data['moves'] != '']
        all_data.reset_index(drop=True, inplace=True)
        return all_data


    """
     _____                                                    
    |  __ \                                                   
    | |__) |_ __  ___  _ __   _ __  ___    ___  ___  ___  ___ 
    |  ___/| '__|/ _ \| '_ \ | '__|/ _ \  / __|/ _ \/ __|/ __|
    | |    | |  |  __/| |_) || |  | (_) || (__|  __/\__ \\__ \
    |_|    |_|   \___|| .__/ |_|   \___/  \___|\___||___/|___/
                      | |                                     
                      |_|                                     
    """


    ###########################
    ### Pretraining Dataset ###
    ###########################

    def pretraining_preprocessing(self, moves, boards):
        buffer_size = len(moves)
        print('buffer_size', buffer_size)
        dataset = tf.data.Dataset.from_tensor_slices(
            (moves, boards)
        )
        dataset = dataset.map(
            self.get_masked_input_and_labels_tf, num_parallel_calls=tf.data.AUTOTUNE
        ).shuffle(buffer_size).prefetch(tf.data.AUTOTUNE)
        return dataset

    def get_masked_input_and_labels(self, inputs, boards):

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
        encoded_texts_masked[inp_mask_2mask] = config.mask_token_id  # mask token is the last in the dict

        # - 10% of masked tokens are replaced with random token
        random_prob = 0.0
        inp_mask_2random = inp_mask_2mask & (np.random.rand(*encoded_texts.shape) < random_prob)
        encoded_texts_masked[inp_mask_2random] = np.random.randint(3, config.mask_token_id, inp_mask_2random.sum())

        # sample_weights: shape(N, 128)
        # - defines the weights for the loss function
        # - weights are 0 for all unmasked token positions and 1 for masked token positions
        sample_weights = np.ones(labels.shape)
        sample_weights[labels == -1] = 0

        # y_labels: shape(N, 128)
        # - defines the labels for the masked tokens
        # - is essentially the input to this function
        y_labels = np.copy(encoded_texts)

        return encoded_texts_masked, y_labels, sample_weights, boards

    def get_masked_input_and_labels_tf(self, inputs, boards):

        # encoded_texts: shape(N, 128) where 128 is the max sequence length
        # - filled with tokenized values
        encoded_texts = inputs

        # inp_mask: shape(N, 128)
        # - filled with True or False denoting whether to mask that token
        # - do not mask special tokens
        inp_mask = tf.random.uniform(encoded_texts.shape) < 0.15
        inp_mask = tf.logical_and(inp_mask, encoded_texts > 2)

        # labels: shape(N, 128)
        # - defines the labels for the masked tokens
        # - all unmasked token positions are set to -1 (ignore)
        labels = -1 * tf.ones(encoded_texts.shape, dtype=tf.int64)
        labels = tf.where(inp_mask, encoded_texts, labels)

        # encoded_texts_masked: shape(N, 128)
        # - modified copy of encoded_texts
        # - masked positions are replaced with mask token id
        encoded_texts_masked = tf.identity(encoded_texts)

        # - 90% of masked tokens are predicted
        predict_prob = 1.0
        inp_mask_2mask = tf.logical_and(inp_mask, tf.random.uniform(encoded_texts.shape) < predict_prob)
        encoded_texts_masked = tf.where(inp_mask_2mask, config.mask_token_id, encoded_texts_masked)

        # - 10% of masked tokens are replaced with random token
        random_prob = 0.0
        inp_mask_2random = tf.logical_and(inp_mask_2mask, tf.random.uniform(encoded_texts.shape) < random_prob)
        random_tokens = tf.random.uniform(inp_mask_2random.shape, minval=3, maxval=config.vocab_size-1, dtype=tf.int64)
        encoded_texts_masked = tf.where(inp_mask_2random, random_tokens, encoded_texts_masked)

        # sample_weights: shape(N, 128)
        # - defines the weights for the loss function
        # - weights are 0 for all unmasked token positions and 1 for masked token positions
        sample_weights = tf.ones(labels.shape, dtype=tf.int32)
        sample_weights = tf.where(labels == -1, 0, sample_weights)

        # y_labels: shape(N, 128)
        # - defines the labels for the masked tokens
        # - is essentially the input to this function
        y_labels = tf.identity(encoded_texts)

        # print(type(encoded_texts_masked), type(y_labels), type(sample_weights), type(boards))
        # print(encoded_texts_masked.shape, y_labels.shape, sample_weights.shape, boards.shape)
        # exit(0)
        return encoded_texts_masked, y_labels, sample_weights, boards





    ####################################
    ### Pretraining Sequence Dataset ###
    ####################################

    def pretraining_sequence_preprocessing(self, moves, boards):
        buffer_size = len(moves)
        print('buffer_size', buffer_size)
        dataset = tf.data.Dataset.from_tensor_slices(
            (moves, boards)
        )
        dataset = dataset.map(
            self.get_masked_seq_input_and_labels_tf2, num_parallel_calls=tf.data.AUTOTUNE
        ).shuffle(buffer_size).prefetch(tf.data.AUTOTUNE)
        return dataset

    def get_masked_seq_input_and_labels(self, inputs, boards):

        # encoded_texts: shape(N, 128) where 128 is the max sequence length
        # - filled with tokenized values
        encoded_texts = inputs

        # 1. Find possible masking positions
        inp_mask = np.random.rand(*encoded_texts.shape) <= 1.0  # Initialize all to True
        inp_mask[encoded_texts <= 2] = False  # Determine which tokens can be masked

        # 2. Find indices where n tokens can be masked in a row
        mask_length = 3
        indices = np.where((inp_mask[:-2] & inp_mask[1:-1] & inp_mask[2:]))[0].tolist()
        mask_start = np.random.choice(indices)  # Choose a random index to start masking
        mask_indices = list(range(mask_start, mask_start + mask_length))

        # 3. Set all entries in inp_mask to False except for the masked indices
        inp_mask[:] = False
        inp_mask[mask_indices] = True

        # 4. Create labels for masked tokens
        labels = -1 * np.ones(encoded_texts.shape, dtype=int)
        labels[inp_mask] = encoded_texts[inp_mask]

        # 5. Create masked input
        encoded_texts_masked = np.copy(encoded_texts)
        encoded_texts_masked[inp_mask] = config.mask_token_id  # mask token is the last in the dict

        # 6. Define loss function weights
        sample_weights = np.ones(labels.shape)
        sample_weights[labels == -1] = 0

        # 7. Finally define labels
        y_labels = np.copy(encoded_texts)

        return encoded_texts_masked, y_labels, sample_weights, boards

    def get_masked_seq_input_and_labels_tf(self, inputs, boards):


        # encoded_texts: shape(N, 128) where 128 is the max sequence length
        # - filled with tokenized values
        encoded_texts = inputs

        # 1. Find possible masking positions
        inp_mask = tf.random.uniform(encoded_texts.shape) <= 1.0  # Initialize all to True
        inp_mask = tf.math.logical_and(inp_mask, encoded_texts > 2)    # Determine which tokens can be masked

        # 2. Find indices where n tokens can be masked in a row
        mask_length = 3
        indices = tf.where(
            tf.math.logical_and(
                tf.math.logical_and(inp_mask[:-2], inp_mask[1:-1]),
                inp_mask[2:]
            )
        )[:, 0]
        mask_start = tf.random.shuffle(indices)[0]  # Choose a random index to start masking
        mask_indices = tf.range(mask_start, mask_start + mask_length)

        # 3. Set all entries in inp_mask to False except for the masked indices
        inp_mask = tf.scatter_nd(tf.expand_dims(mask_indices, 1), tf.ones_like(mask_indices, dtype=tf.bool), inp_mask.shape)

        # 4. Create labels for masked tokens
        labels = -1 * tf.ones(encoded_texts.shape, dtype=tf.int64)
        labels = tf.where(inp_mask, encoded_texts, labels)

        # 5. Create masked input
        encoded_texts_masked = tf.where(inp_mask, config.mask_token_id * tf.ones_like(encoded_texts), encoded_texts)

        # 6. Define loss function weights
        sample_weights = tf.ones(labels.shape)
        sample_weights = tf.where(labels == -1, tf.zeros_like(sample_weights), sample_weights)

        # 7. Finally define labels
        y_labels = tf.identity(encoded_texts)


        # returning encoded_texts_masked causes error
        # returning sample_weights causes error
        return encoded_texts_masked, y_labels, sample_weights, boards




    def get_masked_seq_input_and_labels_tf2(self, inputs, boards):
        # encoded_texts: shape(N, 128) where 128 is the max sequence length
        # - filled with tokenized values
        encoded_texts = inputs

        # 1. Find possible masking positions
        inp_mask = tf.random.uniform(encoded_texts.shape) < 1.0
        inp_mask = tf.logical_and(inp_mask, encoded_texts > 2)

        # 2. Find indices where n tokens can be masked in a row
        mask_length = 3
        shifted_masks = [inp_mask] + [tf.roll(inp_mask, shift=-i, axis=0) for i in range(1, mask_length)]
        combined_mask = tf.reduce_all(shifted_masks, axis=0)
        indices = tf.where(combined_mask)
        indices = tf.reshape(indices, [-1])
        mask_start = tf.random.shuffle(indices)[0]
        mask_indices = tf.range(mask_start, mask_start + mask_length)




        # 3. Set all entries in inp_mask to False except for the masked indices
        inp_mask = tf.scatter_nd(tf.expand_dims(mask_indices, 1), tf.ones_like(mask_indices, dtype=tf.bool),
                                 inp_mask.shape)

        # 4. Create labels for masked tokens
        labels = -1 * tf.ones(encoded_texts.shape, dtype=tf.int64)
        labels = tf.where(inp_mask, encoded_texts, labels)

        # 5. Create masked input
        encoded_texts_masked = tf.identity(encoded_texts)
        mask_token_id = config.mask_token_id
        encoded_texts_masked = tf.where(inp_mask, mask_token_id * tf.ones_like(encoded_texts), encoded_texts)

        # 6. Define loss function weights
        sample_weights = tf.ones(labels.shape, dtype=tf.int64)
        sample_weights = tf.where(tf.equal(labels, -1), tf.zeros_like(labels), sample_weights)

        # 7. Finally define labels
        y_labels = tf.identity(encoded_texts)

        return encoded_texts_masked, y_labels, sample_weights, boards



if __name__ == '__main__':
    dp = DatasetParser()