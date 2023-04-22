# --> Python Imports
import os
import concurrent.futures
import numpy as np
import pickle
import itertools

# --> Tensorflow Imports
from tqdm import tqdm

# --> Threading Imports
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures

# --> Chess Imports
import chess
import chess.pgn as chess_pgn
from preprocess.BoardOperations import BoardOperations

from hydra import config


import threading
import multiprocessing
multiprocessing.set_start_method('fork')


import contextlib

# Define a dummy file-like object that does nothing with its output
class DummyFile:
    def write(self, x):
        pass








class PositionParser:

    def __init__(self):

        # --> Parse Games
        self.parse_dir_games(config.games_file_dir)

    """
     _____                           _____                              
    |  __ \                         / ____|                             
    | |__) |__ _  _ __  ___   ___  | |  __   __ _  _ __ ___    ___  ___ 
    |  ___// _` || '__|/ __| / _ \ | | |_ | / _` || '_ ` _ \  / _ \/ __|
    | |   | (_| || |   \__ \|  __/ | |__| || (_| || | | | | ||  __/\__ \
    |_|    \__,_||_|   |___/ \___|  \_____| \__,_||_| |_| |_| \___||___/
                            
    """

    def parse_dir_games(self, game_dir):
        game_files = os.listdir(game_dir)
        game_dir_name = os.path.basename(os.path.normpath(game_dir))
        save_dir = os.path.join(config.positions_dir, game_dir_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        process_list = []
        for game_file in game_files:
            game_file_path = os.path.join(game_dir, game_file)
            game_file_name = game_file.split('.')[0]
            save_file = os.path.join(save_dir, game_file_name + '.pkl')
            process = multiprocessing.Process(target=self.parse_games_linear, args=(game_file_path, save_file))
            process.start()
            process_list.append(process)
        for th in process_list:
            th.join()

    def parse_games_linear(self, game_file, save_file):
        print('Parsing', game_file, 'to', save_file)
        games = []
        # Iterate over each game,
        with open(game_file) as pgn_file:
            cnt = 0
            while True:
                try:
                    game = chess.pgn.read_game(pgn_file)
                    if game is None:  # End of file
                        break
                    parsed_moves = self.parse_game_moves(game)
                    # print(cnt, parsed_moves)
                    if parsed_moves:
                        games.append(parsed_moves)
                        cnt += 1
                except ValueError as e:
                    continue
        with open(save_file, 'wb') as f:
            pickle.dump(games, f)
        f.close()
        print('Finished parsing', game_file, 'to', save_file)


    """
     _____                                                    
    |  __ \                                                   
    | |__) |_ __  ___  _ __   _ __  ___    ___  ___  ___  ___ 
    |  ___/| '__|/ _ \| '_ \ | '__|/ _ \  / __|/ _ \/ __|/ __|
    | |    | |  |  __/| |_) || |  | (_) || (__|  __/\__ \\__ \
    |_|    |_|   \___|| .__/ |_|   \___/  \___|\___||___/|___/
                      | |                                     
                      |_| 
    - All we ever need to do is store the list of UCI moves...
    - Any additional information can be generated in the training batch preprocessing step
        1. The board tensor can be generated in the preprocessing step 
    """

    def parse_game_moves(self, game):
        moves = ' '.join(list(move.uci() for move in game.mainline_moves()))
        if '@' in moves:
            return None
        else:
            return {'moves': moves}



if __name__ == '__main__':
    pp = PositionParser()

