import chess
import chess.pgn as chess_pgn
from hydra import config
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import os
import chess.engine
from preprocess.BoardOperations import BoardOperations
from preprocess import utils




# --> Get games
games = utils.load_games(config.games_file, max_games=10)
game = games[0]

# --> Get encoded moves
moves_obj = game.mainline_moves()
moves = [move.uci() for move in moves_obj]
moves_str = ' '.join(moves)
encoded_moves = config.encode(moves_str)

# --> Get board tensors
board = chess.Board()
boards_obj = []
for move in moves:
    board.push_uci(move)
    boards_obj.append(board.copy())
board_tensors = [BoardOperations.board_to_tensor(board) for board in boards_obj]

def test_preprocess():
    print('Testing preprocess...')
    print('Board Tensors: ', len(board_tensors))
    print('Moves: ', len(moves))
    print('Encoded Moves: ', len(encoded_moves))

    inputs = encoded_moves
    boards = board_tensors

















if __name__ == '__main__':
    test_preprocess()