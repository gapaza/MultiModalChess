import tensorflow as tf
from keras.layers import TextVectorization
import pandas as pd
import os
import pickle
import time
from hydra import config
from preprocess.strategies.window_masking import rand_window



######################
### Position Limit ###
######################
max_positions_thou = 1





class DatasetParser:

    def __init__(self):
        self.vocab = config.vocab
        self.vocab_size = config.vocab_size

        # --> 1. Get Position Data
        print('--> 1. Get Position Data')
        self.all_positions = self.load_positions_dir(config.positions_load_dir, max_positions_thou)
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
        self.save_datasets()


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
        if len(positions) >= max_positions_thousands * 1000:
            positions = positions[:max_positions_thousands * 1000]
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



    """
      _____                    
     / ____|                   
    | (___    __ _ __   __ ___ 
     \___ \  / _` |\ \ / // _ \
     ____) || (_| | \ V /|  __/
    |_____/  \__,_|  \_/  \___|
                                              
    """

    def save_datasets(self):

        # 1. Derive Filenames
        positions_load_dir_name = os.path.basename(config.positions_load_dir)
        num_positions = int(len(self.all_positions) / 1000)
        num_positions = f"-{num_positions}k"
        time_stamp = time.strftime("%H%M%S")
        train_file_name = positions_load_dir_name + f"-training-{num_positions}-{time_stamp}"
        val_file_name = positions_load_dir_name + f"-validation-{num_positions}-{time_stamp}"

        # 2. Save Datasets
        train_path = os.path.join(config.datasets_dir, train_file_name)
        val_path = os.path.join(config.datasets_dir, val_file_name)
        self.train_dataset.save(train_path)
        self.val_dataset.save(val_path)

        # 3. Print Filenames
        print('Train Dataset: ', train_path)
        print('Val Dataset: ', val_path)





if __name__ == '__main__':
    dp = DatasetParser()