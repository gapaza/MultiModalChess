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

from hydra import config


class PositionParser:

    def __init__(self, max_games=10000):
        self.root_dir = config.root_dir
        self.games_dir = os.path.join(self.root_dir, 'games')
        self.tokens_dir = os.path.join(self.root_dir, 'tokens')
        self.positions_dir = os.path.join(self.root_dir, 'positions')

        # --> All Positions
        # self.games = self.load_games_into_lists(config.games_file, max_games=max_games)
        # self.positions = self.parse_games(self.games)

        # --> Middlegame Positions
        self.positions = self.load_and_parse_middle_games(config.games_file, max_games=max_games)

        # --> Pickle Human Positions
        self.save_positions(self.positions, modifier='middlegame')


    def save_positions(self, positions, modifier=None):
        file_name = '-'.join((config.games_file.split('/')[-1].split('.')[0].split('-')[:-1]))
        if modifier:
            file_name += '-' + modifier
        positions_file_name = file_name + '-positions-' + str(len(positions)) + '.pkl'
        positions_file_path = os.path.join(self.positions_dir, positions_file_name)
        with open(positions_file_path, 'wb') as f:
            print('Saving positions to: ', positions_file_path)
            pickle.dump(positions, f)
            f.close()



    ###################
    ### Middle Game ###
    ###################

    def load_and_parse_middle_games(self, game_file, max_games=10000):
        def parse_game(game):
            parsed_game = self.parse_middle_game_position(game)
            return parsed_game

        print('--> LOADING AND PARSING POSITIONS')
        all_positions = []
        count = 0
        bad_games = 0
        num_threads = 12
        with open(game_file) as pgn_file:
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
                while True:
                    try:
                        game = chess_pgn.read_game(pgn_file)
                        if game is None:  # End of file
                            break
                        future = executor.submit(parse_game, game)
                        all_positions.append(future.result())
                        count += 1
                        if count >= max_games:
                            print('MAX GAMES REACHED')
                            break
                    except ValueError as e:
                        print('--> BAD GAME')
                        bad_games += 1
                        continue

        print('Bad Games: ', bad_games)
        return all_positions

    def parse_middle_games(self, games):
        print('--> PARSING POSITIONS')
        # all_positions = []
        # for game in tqdm(games):
        #     all_positions.append(self.parse_middle_game_position(game))
        # return all_positions

        all_positions = []
        num_threads = 12
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            all_positions = list(tqdm(executor.map(self.parse_middle_game_position, games), total=len(games)))
        return all_positions

    def parse_middle_game_position(self, game):
        board = game.board()
        game_moves = list(move.uci() for move in game.mainline_moves())
        num_moves = len(game_moves)
        half_moves = num_moves // 2
        next_move = None
        half_position = []
        for x in range(half_moves):
            curr_move = game_moves[x]
            next_move = game_moves[x + 1]
            half_position.append(curr_move)
            board.push_uci(curr_move)
        half_board = board.copy()
        half_board_tensor = self.board_to_tensor(half_board)
        position_moves = ' '.join(half_position)
        # print(position_moves)
        # print(half_moves, half_board_tensor.shape, next_move)
        return {
            'moves': position_moves,
            'board': half_board_tensor,
            'next_move': next_move
        }





    #####################
    ### All Positions ###
    #####################

    def load_games_into_lists(self, game_file, max_games=10000):
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

    def parse_games(self, games):
        print('--> PARSING POSITIONS')
        with ThreadPoolExecutor() as executor:
            all_positions = list(tqdm(executor.map(self.parse_game, games), total=len(games)))

        # --> Flatten the list of lists
        all_positions = list(itertools.chain.from_iterable(all_positions))
        return all_positions

    def parse_game(self, game):
        all_moves = [move for move in game.mainline_moves()]
        print('Game Moves: ', len(all_moves))
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
            if len(all_moves) <= (idx + 1):
                break
            board.push(move)
            next_move = all_moves[idx + 1]
            aggregated_moves.append(move.uci())
            move_data = {
                'moves': ' '.join(aggregated_moves),
                'board': self.board_to_tensor(board),
                'next_move': next_move.uci()
            }
            game_data.append(move_data)
        return game_data





    #######################
    ### Board to Tensor ###
    #######################

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




if __name__ == '__main__':
    pp = PositionParser(max_games=1000000)
