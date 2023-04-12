# --> Python Imports
import os
import numpy as np
import pickle
import itertools
from copy import deepcopy

# --> Tensorflow Imports
from tqdm import tqdm
from keras import layers
import tensorflow as tf
from keras_nlp.tokenizers import Tokenizer

# --> Threading Imports
from concurrent.futures import ThreadPoolExecutor

# --> Chess Imports
import chess
import chess.pgn as chess_pgn

from MoveTokenizer import MoveTokenizer


class ChessTokenizers:

    def __init__(self):
        self.root_dir = os.path.dirname(os.path.abspath(__file__))
        self.tokens_dir = os.path.join(self.root_dir, 'tokens')

        # --> Token File
        self.token_file = 'tokens_4672.pkl'
        self.tokens_file = os.path.join(self.tokens_dir, self.token_file)

        # --> Tokenizers
        self.moves_tokenizer = MoveTokenizer(self.token_file)

    def parse_all_tokens(self, game_file, max_games=1000000, save=True):
        possible_moves = set()
        with open(game_file, encoding='utf-8', errors='replace') as pgn_file:
            game_iterator = iter(lambda: chess_pgn.read_game(pgn_file), None)
            for game in tqdm(itertools.islice(game_iterator, max_games), desc="Processing games"):
                try:
                    all_moves = game.mainline_moves()
                    if not any(True for _ in all_moves):
                        continue
                    for move in all_moves:
                        move_uci = move.uci()
                        possible_moves.add(move_uci)
                except ValueError as e:
                    print(e)
                    continue
        if save:
            token_file = os.path.join(self.tokens_dir, 'tokens_' + str(len(possible_moves)) + '.pkl')
            with open(token_file, 'wb') as f:
                pickle.dump(possible_moves, f)
        return possible_moves






if __name__ == '__main__':
    tok = ChessTokenizers()

    # --> Input Tokenizer
    extra_tokens = ['[MASK]']
    input_tokenizer = tok.get_move_tokenizer(100, extra_tokens=extra_tokens)

    print(input_tokenizer.token_to_id("[MASK]"))
    vocabulary = input_tokenizer.get_vocabulary(include_special_tokens=True)

    # --> Output Tokenizer
    output_tokenizer = tok.get_move_tokenizer(1)
