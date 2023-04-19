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

    def __init__(self, max_games=10000, offset=0):
        self.root_dir = config.root_dir
        self.games_dir = os.path.join(self.root_dir, 'games')
        self.tokens_dir = os.path.join(self.root_dir, 'tokens')
        self.positions_dir = os.path.join(self.root_dir, 'positions')

        # --> All Positions
        # self.positions = self.load_and_parse_games(config.games_file, max_games=max_games)

        # --> Middlegame Positions
        self.positions = self.load_and_parse_middle_games2(config.games_file, max_games=max_games, offset=offset)

        # --> Pickle Human Positions
        self.save_positions(self.positions, modifier='middlegame-2')

        # --> Targeted Masking
        # self.load_and_parse_targeted_masking(config.games_file, max_games=max_games)


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

    def save_positions_chunk(self, positions, modifier=None, dir_name='millionbase_full'):
        save_dir = os.path.join(self.positions_dir, dir_name)
        file_name = 'chunk-'
        files = os.listdir(save_dir)
        n = sum([1 for file in files if file.startswith('chunk-')])

        if modifier:
            file_name += '-' + modifier

        positions_file_name = file_name + str(n) + '-positions-' + str(len(positions)) + '.pkl'
        positions_file_path = os.path.join(save_dir, positions_file_name)
        with open(positions_file_path, 'wb') as f:
            print('Saving positions to: ', positions_file_path)
            pickle.dump(positions, f)



    ########################
    ### Targeted Masking ###
    ########################

    def load_and_parse_targeted_masking(self, game_file, max_games=10000):
        def parse_game(game):
            parsed_game = self.parse_targeted_game(game)
            return parsed_game

        def parse_chunk(chunk):
            parsed_chunk = [parse_game(game) for game in chunk]
            return parsed_chunk

        def chunk_games(games_list, chunk_size):
            for i in range(0, len(games_list), chunk_size):
                yield games_list[i:i + chunk_size]

        print('--> LOADING AND PARSING POSITIONS')
        all_positions = []
        bad_games = 0
        num_threads = 24

        with open(game_file) as pgn_file:
            games = []
            count = 0
            progress_bar = tqdm(range(max_games), desc="Loading games", unit="game")
            for _ in progress_bar:
                try:
                    game = chess.pgn.read_game(pgn_file)
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
            progress_bar.close()

        chunk_size = len(games) // num_threads

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            chunked_games = list(chunk_games(games, chunk_size))
            progress_bar = tqdm(total=len(games), desc="Processing games", unit="game")
            for results in executor.map(parse_chunk, chunked_games):
                for result in results:
                    all_positions.append(result)
                    progress_bar.update(1)
                    save_point = 10000
                    if len(all_positions) % save_point == 0:
                        print('Saving positions...')
                        # modifier is targeted-masking and the size of all_positions - save_point
                        self.save_positions_chunk(all_positions, modifier='targeted-masking')
                        all_positions = []


            progress_bar.close()

        print('Bad Games: ', bad_games)

    def parse_targeted_game(self, game):
        board = game.board()
        game_moves = list(move.uci() for move in game.mainline_moves())
        num_moves = len(game_moves)
        board_tensors = []
        for x in range(num_moves):
            curr_move = game_moves[x]
            board.push_uci(curr_move)
            board_tensor = self.board_to_tensor(board)
            board_tensors.append(board_tensor)

        return {
            'moves': ' '.join(game_moves),
            'boards': board_tensors,
        }







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
        num_threads = 32
        with open(game_file) as pgn_file:
            progress_bar = tqdm(range(max_games), desc="Processing games", unit="game")
            for _ in progress_bar:
                try:
                    game = chess_pgn.read_game(pgn_file)
                    if game is None:  # End of file
                        break
                    result = parse_game(game)
                    all_positions.append(result)
                    count += 1
                    if count >= max_games:
                        print('MAX GAMES REACHED')
                        break
                except ValueError as e:
                    print('--> BAD GAME')
                    bad_games += 1
                    continue
            progress_bar.close()

        print('Bad Games: ', bad_games)
        return all_positions

    def load_and_parse_middle_games2(self, game_file, max_games=10000, offset=0):
        def parse_game(game):
            parsed_game = self.parse_middle_game_position(game)
            return parsed_game

        def parse_chunk(chunk):
            parsed_chunk = [parse_game(game) for game in chunk]
            return parsed_chunk

        def chunk_games(games_list, chunk_size):
            for i in range(0, len(games_list), chunk_size):
                yield games_list[i:i + chunk_size]

        print('--> LOADING AND PARSING POSITIONS')
        all_positions = []
        bad_games = 0
        num_threads = 24

        with open(game_file) as pgn_file:
            for i in tqdm(range(offset), desc="Applying offset", unit="game"):
                try:
                    game = chess.pgn.read_game(pgn_file)
                    if game is None:  # End of file
                        break
                except ValueError as e:
                    continue

            games = []
            count = 0
            progress_bar = tqdm(range(max_games), desc="Loading games", unit="game")
            for _ in progress_bar:
                try:
                    game = chess.pgn.read_game(pgn_file)
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
            progress_bar.close()

        chunk_size = len(games) // num_threads

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            chunked_games = list(chunk_games(games, chunk_size))
            progress_bar = tqdm(total=len(games), desc="Processing games", unit="game")
            for results in executor.map(parse_chunk, chunked_games):
                for result in results:
                    all_positions.append(result)
                    progress_bar.update(1)
            progress_bar.close()
        return all_positions

    def load_and_parse_middle_games3(self, game_file, batch_size=1000000):
        def parse_game(game):
            parsed_game = self.parse_middle_game_position(game)
            return parsed_game

        def parse_chunk(chunk):
            parsed_chunk = [parse_game(game) for game in chunk]
            return parsed_chunk

        def chunk_games(games_list, chunk_size):
            for i in range(0, len(games_list), chunk_size):
                yield games_list[i:i + chunk_size]

        def save_batch(batch_data, batch_number):
            file_name = f"parsed_games_batch_{batch_number}.pkl"
            with open(file_name, "wb") as f:
                pickle.dump(batch_data, f)
            print(f"Saved batch {batch_number} to {file_name}")

        print('--> LOADING AND PARSING POSITIONS')
        num_threads = 24
        chunk_size = batch_size // num_threads

        with open(game_file) as pgn_file:
            batch_number = 1
            while True:
                games = []
                for _ in range(batch_size):
                    try:
                        game = chess.pgn.read_game(pgn_file)
                        if game is None:  # End of file
                            break
                        games.append(game)
                    except ValueError as e:
                        print('--> BAD GAME')
                        continue

                if not games:
                    break

                with ThreadPoolExecutor(max_workers=num_threads) as executor:
                    chunked_games = list(chunk_games(games, chunk_size))
                    all_positions = []
                    progress_bar = tqdm(total=len(games), desc=f"Processing batch {batch_number} games", unit="game")
                    for results in executor.map(parse_chunk, chunked_games):
                        for result in results:
                            all_positions.append(result)
                            progress_bar.update(1)
                    progress_bar.close()

                    save_batch(all_positions, batch_number)
                batch_number += 1

        print('Finished processing all games')


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
        return {
            'moves': position_moves,
            'board': half_board_tensor,
            'next_move': next_move
        }



    #####################
    ### All Positions ###
    #####################


    def load_and_parse_games(self, game_file, max_games=10000):
        games = self.load_games_into_lists(game_file, max_games=max_games)
        with ThreadPoolExecutor(max_workers=64) as executor:
            all_positions = list(tqdm(executor.map(self.parse_game, games), total=len(games)))

        # --> Flatten the list of lists
        all_positions = list(itertools.chain.from_iterable(all_positions))
        return all_positions

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
    pp = PositionParser(max_games=1000000, offset=1000000)
