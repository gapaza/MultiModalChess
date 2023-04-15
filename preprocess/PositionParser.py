# --> Python Imports
import os
import numpy as np
import pickle
import itertools

# --> Tensorflow Imports
from tqdm import tqdm

# --> Threading Imports
from concurrent.futures import ThreadPoolExecutor

# --> Chess Imports
import chess
import chess.pgn as chess_pgn

from hydra import config


class PositionParser:

    def __init__(self, max_games=10000):
        self.root_dir = config.root_dir
        self.games_dir = os.path.join(self.root_dir, 'games')
        self.tokens_dir = os.path.join(self.root_dir, 'tokens')
        self.positions_dir = os.path.join(self.root_dir, 'positions')

        # --> Parse Human Games, Positions
        self.human_games_file = os.path.join(self.games_dir, 'human-training-games.pgn')
        self.human_games = self.parse_games(self.human_games_file, max_games=max_games)
        self.human_positions = self.parse_positions(self.human_games)

        # --> Pickle Human Positions
        self.human_positions_file = os.path.join(self.positions_dir, 'human-training-positions-' + str(len(self.human_positions)) + '.pkl')
        with open(self.human_positions_file, 'wb') as f:
            pickle.dump(self.human_positions, f)

    def parse_games(self, game_file, max_games=10000):
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

    def parse_positions(self, gameset):
        print('--> PARSING POSITIONS')
        with ThreadPoolExecutor() as executor:
            all_games = list(tqdm(executor.map(self.parse_position_fast, gameset), total=len(gameset)))

        # --> Flatten the list of lists
        all_positions = list(itertools.chain.from_iterable(all_games))
        return all_positions

    def parse_position_fast(self, game):
        all_moves = [move for move in game.mainline_moves()]
        if len(all_moves) == 0:
            return []

        # --> Initialize Data
        board = game.board()
        game_data = []
        aggregated_moves = []

        # --> First Move
        first_move = all_moves[0].uci()
        sequence_data = ' '.join(aggregated_moves)
        move_data = {
            'moves': sequence_data,
            'board': self.board_to_tensor(board),
            'next_move': first_move
        }
        game_data.append(move_data)

        # --> Other Moves
        for idx, move in enumerate(all_moves):
            board.push(move)
            if len(all_moves) <= (idx + 1):
                break
            next_move = all_moves[idx + 1]
            aggregated_moves.append(move.uci())
            move_data = {
                'moves': ' '.join(aggregated_moves),
                'board': self.board_to_tensor(board),
                'next_move': next_move.uci()
            }
            game_data.append(move_data)
        return game_data

    def board_to_tensor(self, board):
        tensor = np.zeros((8, 8, 12))
        for square, piece in board.piece_map().items():
            rank, file = divmod(square, 8)
            piece_index = self.piece_to_index(piece)
            tensor[7 - rank, file, piece_index] = 1
        return tensor

    def piece_to_index(self, piece):
        piece_order = ['P', 'N', 'B', 'R', 'Q', 'K']
        index = piece_order.index(piece.symbol().upper())
        # If the piece is black, add 6 to the index (to cover the range 0-11)
        if piece.color == chess.BLACK:
            index += 6
        return index

    def split_data(self, all_games):
        # Calculate the number of samples for train and validation sets
        num_total_samples = len(all_games)
        num_val_samples = int(0.15 * num_total_samples)
        num_train_samples = num_total_samples - num_val_samples
        # Split the data into train and validation sets
        train_set = all_games[:num_train_samples]
        val_set = all_games[num_train_samples:]
        return train_set, val_set



if __name__ == '__main__':
    pp = PositionParser(max_games=10)
