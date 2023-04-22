import tensorflow as tf
from keras.layers import TextVectorization
import pandas as pd
import os
import pickle
import time
from hydra import config
from preprocess.strategies.window_masking import rand_window

class MoveFileParser:

    def __init__(self):

        # 1. Get positions directory
        print('Parsing:', config.positions_load_dir)
        self.parse_dir = config.positions_load_dir
        self.time_stamp = time.strftime("%H%M%S")

        # 2. Get move files
        move_files = self.load_move_files()
        print('Move Files:', move_files)

        # 3. Parse move files
        dataset = tf.data.Dataset.from_tensor_slices(move_files)
        dataset = dataset.interleave(
            lambda x: tf.data.TextLineDataset(x).map(config.encode_tf, num_parallel_calls=tf.data.AUTOTUNE).map(
                rand_window, num_parallel_calls=tf.data.AUTOTUNE
            ).prefetch(tf.data.AUTOTUNE),
            cycle_length=tf.data.AUTOTUNE,
            block_length=16,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False
        )

        # 4. Save dataset
        print('Saving dataset...')
        self.save_dataset(dataset)


    def save_dataset(self, dataset, label='train'):
        positions_load_dir_name = os.path.basename(config.positions_load_dir)
        file_name = positions_load_dir_name + f"-{label}-{self.time_stamp}"
        file_path = os.path.join(config.datasets_dir, file_name)
        dataset.save(file_path)
        print('Dataset: ', label, file_path)




    def load_move_files(self):
        move_files = []
        for file in os.listdir(self.parse_dir):
            if file.endswith('.txt'):
                full_path = os.path.join(self.parse_dir, file)
                move_files.append(full_path)

        return move_files









if __name__ == '__main__':
    mfp = MoveFileParser()



