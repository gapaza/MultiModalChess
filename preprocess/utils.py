import chess
import chess.pgn as chess_pgn
from hydra import config
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import os
import chess.engine

def load_games(game_file, max_games=10000):
    games = []
    count = 0
    bad_games = 0
    with open(game_file) as pgn_file:
        while True:
            try:
                game = chess_pgn.read_game(pgn_file)
                if game is None:  # End of file
                    break
                games.append(game)
                count += 1
                if count >= max_games:
                    print('MAX GAMES REACHED')
                    break
            except ValueError as e:
                print('--> BAD GAME')
                bad_games += 1
                continue
    print('Bad Games: ', bad_games)
    return games



