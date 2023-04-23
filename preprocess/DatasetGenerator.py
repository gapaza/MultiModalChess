import tensorflow as tf
from keras.layers import TextVectorization
import pandas as pd
import os
import pickle
import time
from hydra import config
from preprocess.strategies.window_masking import rand_window, rand_window_batch

class DatasetGenerator:

    def __init__(self):

        # 1. Get positions directory
        print('Parsing:', config.positions_load_dir)
        self.parse_dir = config.positions_load_dir
        self.time_stamp = time.strftime("%H%M%S")

        # 2. Get move files
        move_files = self.load_move_files()
        print('Move Files:', move_files)

        # 3. Split files with 90% train and 10% validation
        split_idx = int(len(move_files) * 0.9)
        train_move_files, val_move_files = move_files[:split_idx], move_files[split_idx:]
        print('Train Move Files:', train_move_files)
        print('Val Move Files:', val_move_files)

        # 3. Parse move files
        self.train_dataset = self.parse_dataset(train_move_files)
        self.val_dataset = self.parse_dataset(val_move_files)

        # 4. Save dataset
        if config.save_dataset:
            print('Saving dataset...')
            self.save_dataset(self.train_dataset, label='train')
            self.save_dataset(self.val_dataset, label='val')


    def parse_dataset(self, move_files):
        def parse_fn(file_path):
            dataset = tf.data.TextLineDataset(file_path)

            # Normal Config
            # dataset = dataset.map(config.encode_tf, num_parallel_calls=tf.data.AUTOTUNE)
            # dataset = dataset.map(rand_window, num_parallel_calls=tf.data.AUTOTUNE)

            # Early Batching
            dataset = dataset.batch(config.batch_size)
            dataset = dataset.map(config.encode_tf_batch, num_parallel_calls=tf.data.AUTOTUNE)
            dataset = dataset.map(rand_window_batch, num_parallel_calls=tf.data.AUTOTUNE)
            dataset = dataset.shuffle(10)
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



    def get_datasets(self):
        return self.train_dataset, self.val_dataset

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
    dg = DatasetGenerator()
    train_dataset, val_dataset = dg.get_datasets()
    count = 0
    print('\n\n -- TESTING -- \n\n')
    print('train_dataset:', train_dataset)
    print('train_dataset elements:', train_dataset.element_spec)




    # _ParallelInterleaveDataset
    # _PrefetchDataset
    for out1 in train_dataset.take(1):
        print('count:', count)
        count += 1
        print('out1:', out1)







