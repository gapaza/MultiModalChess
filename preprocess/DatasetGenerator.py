import tensorflow as tf
from keras.layers import TextVectorization
import pandas as pd
import os
import pickle
import time
from hydra import config
from tqdm import tqdm
from preprocess.strategies.window_masking import rand_window, rand_window_batch

class DatasetGenerator:

    def __init__(self):

        # 1. Get evaluations directory
        print('Parsing:', config.stockfish_data_file)
        self.parse_dir = config.stockfish_data_dir
        self.time_stamp = time.strftime("%H%M%S")

        # 2. Load Moves
        # load pickle file
        self.stockfish_data = pickle.load(open(config.stockfish_data_file, 'rb'))




        move_files = self.load_move_files()
        print('Move Files:', move_files)

        # 3. Split files with 90% train and 10% validation
        split_idx = int(len(move_files) * 0.9)
        self.train_move_files, self.val_move_files = move_files[:split_idx], move_files[split_idx:]
        print('Train Move Files:', self.train_move_files)
        print('Val Move Files:', self.val_move_files)




    def get_interleave_dataset(self, save=False):
        train_dataset = self.parse_interleave_dataset(self.train_move_files)
        val_dataset = self.parse_interleave_dataset(self.val_move_files)
        if save:
            self.save_dataset(train_dataset, label='train-interleave')
            self.save_dataset(val_dataset, label='val-interleave')
        return train_dataset, val_dataset

    def parse_interleave_dataset(self, move_files):
        def parse_fn(file_path):
            dataset = tf.data.TextLineDataset(file_path)

            # Normal Config
            # dataset = dataset.map(config.encode_tf, num_parallel_calls=tf.data.AUTOTUNE)
            # dataset = dataset.map(rand_window, num_parallel_calls=tf.data.AUTOTUNE)

            # Early Batching
            dataset = dataset.batch(config.batch_size)
            dataset = dataset.map(config.encode_tf_batch, num_parallel_calls=tf.data.AUTOTUNE)
            dataset = dataset.map(rand_window_batch, num_parallel_calls=tf.data.AUTOTUNE)
            # dataset = dataset.shuffle(3125)
            return dataset.prefetch(tf.data.AUTOTUNE)

        dataset = tf.data.Dataset.from_tensor_slices(move_files)
        dataset = dataset.interleave(
            parse_fn,
            cycle_length=tf.data.AUTOTUNE,
            block_length=1,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=True
        )
        dataset = dataset.shuffle(50)
        return dataset.prefetch(tf.data.AUTOTUNE)



    def get_memory_dataset(self, save=False):
        train_dataset = self.parse_memory_dataset(self.train_move_files)
        val_dataset = self.parse_memory_dataset(self.val_move_files)
        if save:
            self.save_dataset(train_dataset, label='train-memory')
            self.save_dataset(val_dataset, label='val-memory')
        return train_dataset, val_dataset

    def parse_memory_dataset(self, move_files):
        print(move_files)

        full_dataset = tf.data.TextLineDataset(move_files)
        full_dataset = full_dataset.batch(config.batch_size)
        full_dataset = full_dataset.map(config.encode_tf_batch, num_parallel_calls=tf.data.AUTOTUNE)
        full_dataset = full_dataset.map(rand_window_batch, num_parallel_calls=tf.data.AUTOTUNE)
        full_dataset = full_dataset.shuffle(100)
        return full_dataset.prefetch(tf.data.AUTOTUNE)





    def save_dataset(self, dataset, label='train'):
        positions_load_dir_name = os.path.basename(config.positions_load_dir)
        file_name = positions_load_dir_name + f"-{label}-{self.time_stamp}"
        file_path = os.path.join(config.datasets_dir, file_name)
        dataset.save(file_path)
        print('Dataset: ', label, file_path)



    @staticmethod
    def load_datasets():
        train_dataset_path = os.path.join(config.datasets_dir, config.train_dataset)
        val_dataset_path = os.path.join(config.datasets_dir, config.val_dataset)
        train_dataset = tf.data.Dataset.load(train_dataset_path)
        val_dataset = tf.data.Dataset.load(val_dataset_path)
        return train_dataset, val_dataset





if __name__ == '__main__':
    dg = DatasetGenerator()
    # dg.save_datasets()


    print('\n\n -- TESTING -- \n\n')
    train_dataset, val_dataset = dg.get_memory_dataset()
    count = 0
    for out1, out2, out3, out4 in train_dataset.take(1):
        print('out1:', out1)
        print('out2:', out2)
        print('out3:', out3)
        print('out4:', out4)







