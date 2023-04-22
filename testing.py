import chess
from hydra import config
import tensorflow as tf
import chess.engine
from preprocess import utils
import os


def get_game():
    game_file = os.path.join(config.games_dir, 'human-training-games.pgn')
    games = utils.load_games(game_file, max_games=10)
    game = games[0]
    return game


def get_encoded_game_moves():
    game_file = os.path.join(config.games_dir, 'human-training-games.pgn')
    games = utils.load_games(game_file, max_games=10)
    game = games[0]
    moves = [move.uci() for move in game.mainline_moves()]
    encoded_moves = config.tokenizer(' '.join(moves))
    return moves, encoded_moves






##################
### Strategies ###
##################

from preprocess.strategies.window_masking import rand_window
from preprocess.strategies.window_masking import rand_window_rand_game_token



def test_strategy():
    moves, encoded_moves = get_encoded_game_moves()
    print('Moves:', moves)
    print('Encoded Moves:', encoded_moves)

    encoded_texts_masked, y_labels, sample_weights, board_tensor = rand_window(encoded_moves)



####################
### Utility Test ###
####################

def test_utility():
    game = get_game()



    uci_moves = [move.uci() for move in game.mainline_moves()]

    # parse san moves
    san_moves = []
    board = chess.Board()
    for move in game.mainline_moves():
        san_moves.append(board.san(move))
        board.push(move)





    print(uci_moves)
    print(san_moves)




if __name__ == '__main__':
    print('Testing Strategy')
    test_utility()

