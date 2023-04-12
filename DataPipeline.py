# --> Python Imports
import os
import json
from copy import deepcopy
import numpy as np
import pickle
import itertools

# --> Tensorflow Imports
import tensorflow as tf
from tqdm import tqdm
from keras_nlp.layers import MaskedLMMaskGenerator

# --> Threading Imports
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Thread
from queue import Queue

# --> Chess Imports
import chess
import chess.pgn as chess_pgn
from chess.pgn import Game, Mainline
from chess import Board, Move

from ChessTokenizers import ChessTokenizers



class DataPipeline:

    def __init__(self):
        self.root_dir = os.path.dirname(os.path.abspath(__file__))
        self.games_dir = os.path.join(self.root_dir, 'games')
        self.tokens_dir = os.path.join(self.root_dir, 'tokens')
        self.positions_dir = os.path.join(self.root_dir, 'positions')
        self.datasets_dir = os.path.join(self.root_dir, 'datasets')

        # --> Tokenizer
        self.tokenizer_manager = ChessTokenizers()
        self.tokenizer = self.tokenizer_manager.moves_tokenizer

        # --> Create Mask Generator
        self.mask_layer = MaskedLMMaskGenerator(
            vocabulary_size=self.tokenizer.vocabulary_size(),
            mask_selection_rate=0.15,
            mask_selection_length=1,
            mask_token_id=self.tokenizer.token_to_id("[MASK]"),
        )

        # --> Read positions file
        self.human_positions_file = os.path.join(self.positions_dir, 'human-training-positions-627.pkl')
        self.human_positions = []
        with open(self.human_positions_file, 'rb') as f:
            self.human_positions = pickle.load(f)



    def pretraining_preprocess(self, board, moves, next_move):
        # --> Preprocess Moves for MLM
        encoded_moves = self.tokenizer.encode(moves)
        masked_moves = self.mask_layer(encoded_moves)
        features = {
            "token_ids": masked_moves["token_ids"],
            "mask_positions": masked_moves["mask_positions"],
        }
        labels = masked_moves["mask_ids"]
        weights = masked_moves["mask_weights"]
        return features, labels, weights
        # return ({"moves": encoded_moves, "board": board}, next_move)
        # return ({"moves": encoded_moves, "board": board}, next_move)


    def parse_dataset(self, preprocess, batch_size=32):

        # --> Iterate over all positions
        all_boards = []
        all_moves = []
        all_next_moves = []
        for position in tqdm(self.human_positions, desc='Parsing positions into tensors'):
            all_boards.append(position['board'])
            all_moves.append(position['moves'])
            all_next_moves.append(position['next_move'])

        dataset = tf.data.Dataset.from_tensor_slices((all_boards, all_moves, all_next_moves))
        dataset = dataset.batch(batch_size)
        dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        return dataset.shuffle(2048).prefetch(16).cache()






    def parse_training_dataset(self):
        return 0














