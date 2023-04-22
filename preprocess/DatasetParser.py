import tensorflow as tf
from keras.layers import TextVectorization
import pandas as pd
import numpy as np
import os
import pickle
import re
import shutil

from testing import get_board_tensor_from_moves


from preprocess.strategies.window_masking import rand_window







from hydra import config
import sys



class DatasetParser:

    def __init__(self):
        self.vocab = config.vocab
        self.vocab_size = config.vocab_size

        # --> 1. Get Position Data
        print('--> 1. Get Position Data')
        self.all_positions = self.load_positions_dir(config.positions_load_dir)
        self.all_positions = self.all_positions.sample(frac=1).reset_index(drop=True)

        # --> 2. Encode Moves and Extract Boards
        print('--> 2. Encode Moves and Extract Boards')
        self.all_moves = config.encode(self.all_positions.moves.values.tolist())

        # --> 3. Split into Train and Validation
        print('--> 3. Split into Train and Validation')
        split_idx = int(len(self.all_moves) * 0.9)
        self.train_moves, self.validation_moves = self.all_moves[:split_idx],  self.all_moves[split_idx:]

        # --> 4. Preprocess Train and Validation Datasets
        print('--> 4. Preprocess Train and Validation Datasets')
        self.train_dataset = self.preprocess(self.train_moves, rand_window)
        self.val_dataset = self.preprocess(self.validation_moves, rand_window)


        # --> 5. Save Datasets
        print('--> 5. Save Datasets')
        positions_load_dir_name = os.path.basename(config.positions_load_dir)
        num_positions = int(len(self.all_positions) / 1000)
        num_positions = f"-{num_positions}k"
        train_path = os.path.join(config.datasets_dir, positions_load_dir_name + '-training' + num_positions)
        val_path = os.path.join(config.datasets_dir, positions_load_dir_name + '-validation' + num_positions)
        self.train_dataset.save(train_path)
        print('Train Dataset: ', train_path)
        self.val_dataset.save(val_path)
        print('Val Dataset: ', val_path)



    """
     _                        _   _____             _  _    _                    
    | |                      | | |  __ \           (_)| |  (_)                   
    | |      ___    __ _   __| | | |__) |___   ___  _ | |_  _   ___   _ __   ___ 
    | |     / _ \  / _` | / _` | |  ___// _ \ / __|| || __|| | / _ \ | '_ \ / __|
    | |____| (_) || (_| || (_| | | |   | (_) |\__ \| || |_ | || (_) || | | |\__ \
    |______|\___/  \__,_| \__,_| |_|    \___/ |___/|_| \__||_| \___/ |_| |_||___/
                                                                                      
    """

    def load_positions_dir(self, positions_dir, max_positions_thousands=10000):
        # Iterate and open all files in dir with pickle
        positions = []
        for idx, filename in enumerate(os.listdir(positions_dir)):
            print('--> Load Positions: ', filename)
            with open(os.path.join(positions_dir, filename), 'rb') as f:
                positions += pickle.load(f)
            f.close()
            if len(positions) >= max_positions_thousands * 1000:
                break
        print('Total Positions: ', len(positions))
        all_data = pd.DataFrame(positions)
        all_data = self.prune_positions(all_data)
        return all_data

    def prune_positions(self, all_data):
        # --> Conditions for dropping positions
        # 1. No moves exist
        # all_data = all_data[all_data['moves'] != '']
        all_data = all_data[all_data['moves'].apply(lambda x: len(x) >= 30)]
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

    def preprocess(self, encoded_moves, preprocess_func):
        full_buffer = len(encoded_moves)
        dataset = tf.data.Dataset.from_tensor_slices(
            (encoded_moves)
        )
        dataset = dataset.map(
            preprocess_func, num_parallel_calls=tf.data.AUTOTUNE
        ).shuffle(1000000).prefetch(tf.data.AUTOTUNE)
        return dataset





    ###########################
    ### Pretraining Dataset ###
    ###########################

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

    def get_masked_seq_input_and_labels_tf(self, inputs, board):
        # encoded_texts: shape(128,) where 128 is the max sequence length
        # - filled with tokenized values
        encoded_texts = inputs

        # 1. Find possible masking positions
        # inp_mask.shape: (128,)
        inp_mask = tf.random.uniform(encoded_texts.shape) <= 1.0
        inp_mask = tf.logical_and(inp_mask, encoded_texts > 2)

        # 2. Find indices where n tokens can be masked in a row
        true_indices = tf.squeeze(tf.where(inp_mask), axis=1)
        rand_idx = tf.random.uniform(shape=[], maxval=tf.shape(true_indices)[0], dtype=tf.int32)
        mask_start = tf.gather(true_indices, rand_idx)
        mask_length = 3
        mask_length = tf.minimum(tf.cast(mask_length, dtype=tf.int64), 128 - mask_start)
        mask_indices = tf.range(mask_start, mask_start + mask_length)

        # 3. Set all entries in inp_mask to False except for the masked indices
        inp_mask = tf.zeros((128,), dtype=tf.bool)
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

        return encoded_texts_masked, y_labels, sample_weights, board




    ######################################################
    ### Pretraining Position Targeted Sequence Dataset ###
    ######################################################

    def get_masked_seq_input_and_labels_position_targeted(self, inputs, board, mask_pos):
        encoded_texts = inputs

        # 1. Find possible masking positions
        # inp_mask.shape: (128,)
        inp_mask = tf.random.uniform(encoded_texts.shape) <= 1.0
        inp_mask = tf.logical_and(inp_mask, encoded_texts > 2)

        # --> Mask of length 3
        mask_length = 3
        mask_start = mask_pos - 1
        mask_indices = tf.range(mask_start, mask_start + mask_length)

        # --> Mask of length 5
        # mask_length = 5
        # mask_start = mask_pos - 2
        # mask_indices = tf.range(mask_start, mask_start + mask_length)

        # 3. Set all entries in inp_mask to False except for the masked indices
        inp_mask = tf.zeros((128,), dtype=tf.bool)
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

        return encoded_texts_masked, y_labels, sample_weights, board


        return 0




    ##########################
    ### Pretraining Custom ###
    ##########################

    def pretraining_sequence_preprocessing_custom(self, encoded_texts):
        print('Custom Preprocess')

        # 1. Find possible masking positions
        # inp_mask.shape: (128,)
        inp_mask = tf.random.uniform(encoded_texts.shape) <= 1.0
        inp_mask = tf.logical_and(inp_mask, encoded_texts > 2)

        true_indices = tf.where(inp_mask)
        first_true_index = true_indices[0]
        inp_mask = tf.concat([
            inp_mask[:first_true_index[0]],
            [False],
            inp_mask[first_true_index[0] + 1:]
        ], axis=0)

        last_true_index = true_indices[-1]
        inp_mask = tf.concat([
            inp_mask[:last_true_index[0]],
            [False],
            inp_mask[last_true_index[0] + 1:]
        ], axis=0)
        sequence_end = last_true_index[0] - 2

        # 1.1 Get board tensor using tf.py_function to call get_board_tensor_from_moves
        board_tensor = tf.py_function(get_board_tensor_from_moves, [encoded_texts, sequence_end], tf.int64)
        board_tensor.set_shape((8, 8, 12))

        # 2. Find center point where 3 tokens can be masked by a mask slide
        true_indices = tf.squeeze(tf.where(inp_mask), axis=1)
        rand_idx = tf.random.uniform(shape=[], maxval=tf.shape(true_indices)[0], dtype=tf.int32)
        mask_center = tf.gather(true_indices, rand_idx)
        mask_start = mask_center - 1
        encoded_texts_cutoff = mask_start + 3
        mask_length = 3
        mask_indices = tf.range(mask_start, mask_start + mask_length)

        # Set all tokens in encoded_text to config.padding_token_id after cutoff
        # encoded_texts = tf.concat([
        #     encoded_texts[:encoded_texts_cutoff],
        #     config.padding_token_id * tf.ones_like(encoded_texts[mask_start + mask_length:])
        # ], axis=0)
        # encoded_texts.set_shape((128,))

        # 3. Set all entries in inp_mask to False except for the masked indices
        inp_mask = tf.zeros((128,), dtype=tf.bool)
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

        # 8. Print all returns
        print('Encoded Texts Masked: ', encoded_texts_masked)
        print('Sample Weights: ', sample_weights)
        print('Y Labels: ', y_labels)
        # print('Board: ', board)

        return encoded_texts_masked, y_labels, sample_weights, board_tensor


if __name__ == '__main__':
    dp = DatasetParser()