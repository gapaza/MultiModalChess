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


def board_to_tensor(board):
    tensor = np.zeros((8, 8, 12))
    for square, piece in board.piece_map().items():
        rank, file = divmod(square, 8)
        piece_index = piece_to_index(piece)
        tensor[7 - rank, file, piece_index] = 1
    return tensor

def piece_to_index(piece):
    piece_order = ['P', 'N', 'B', 'R', 'Q', 'K']
    index = piece_order.index(piece.symbol().upper())
    # If the piece is black, add 6 to the index (to cover the range 0-11)
    if piece.color == chess.BLACK:
        index += 6
    return index



def test_engine():
    import chess.engine
    stockfish_path = '/opt/homebrew/Cellar/stockfish/15.1/bin/stockfish'
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    print(engine.id)

    # Set the maximum number of games to process
    max_games = 1

    # Open the PGN file
    with open(config.games_file) as pgn_file:
        progress_bar = tqdm(range(max_games), desc="Processing games", unit="game")
        for _ in progress_bar:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            game_moves = list(move for move in game.mainline_moves())
            game_moves_uci = list(move.uci() for move in game_moves)
            middle_game = len(game_moves) // 2
            board = game.board()
            next_move = game_moves[0]
            for x in range(middle_game):
                move = game_moves[x]
                next_move = game_moves[x + 1]
                board.push(move)
            print('Analyzing board')
            info = engine.analyse(board, chess.engine.Limit(time=5), multipv=5)
            for idx, variation in enumerate(info, start=1):
                move = variation.get("pv")[0]
                score = variation.get("score")
                if board.turn == chess.BLACK:
                    score = score.relative.score()
                print(f"{idx}. {move} ({score}) {middle_game}")





        progress_bar.close()
        engine.quit()





def test_chess():


    games = load_games(config.games_file, max_games=10)
    print(len(games))
    positions = []
    for game in games:
        board = game.board()
        game_moves = list(move.uci() for move in game.mainline_moves())



        print(game_moves)
        exit(0)
        num_moves = len(game_moves)
        half_moves = num_moves // 2
        next_move = None
        half_position = []
        for x in range(half_moves):
            curr_move = game_moves[x]
            next_move = game_moves[x + 1]
            half_position.append(curr_move)
            board.push_uci(curr_move)
            positions.append(game_moves[x])
        half_board = board.copy()
        half_board_tensor = board_to_tensor(half_board)
        position_moves = ' '.join(half_position)
        print(position_moves)
        print(half_moves, half_board_tensor.shape, next_move)
        positions.append({
            'moves': position_moves,
            'board': half_board_tensor,
            'next_move': next_move
        })



from hydra import config



def test_encoding():
    move_sequence = ['a1a2', 'd5c4', 'c2c4', 'g7g6', 'b1c3', 'd7d5', 'c1f4', 'f8g7', 'f4e5', 'd5c4', 'e2e3', 'b8c6', 'd1a4', 'e8g8', 'e5f6', 'g7f6', 'f1c4', 'a7a6', 'c4d5', 'b7b5', 'a4d1', 'c8b7', 'a2a3', 'e7e6', 'd5f3', 'c6a5', 'f3b7', 'a5b7', 'b2b4']
    last_half = ['c7c5', 'b4c5', 'b7c5', 'g1f3', 'd8a5', 'd1c2', 'c5a4', 'a1c1', 'a8c8', 'e1g1', 'a5c3', 'c2e2', 'c3a3', 'c1c2', 'c8c2', 'e2c2', 'a3c3', 'c2e4', 'f8c8', 'g2g3', 'c3c2', 'e4b7', 'c2c6']
    boards = None

    move_string = ' '.join(move_sequence)
    encoded_texts = config.encode(move_string)
    print(move_string)
    print(encoded_texts)

    # 1. Find possible masking positions
    inp_mask = np.random.rand(*encoded_texts.shape) < 10  # Initialize all to True
    inp_mask[encoded_texts <= 2] = False                  # Determine which tokens can be masked

    # 2. Find indices where n tokens can be masked in a row
    mask_length = 3
    indices = np.where((inp_mask[:-2] & inp_mask[1:-1] & inp_mask[2:]))[0].tolist()
    mask_start = np.random.choice(indices)  # Choose a random index to start masking
    mask_indices = list(range(mask_start, mask_start + mask_length))
    print(mask_indices)

    # 3. Set all entries in inp_mask to False except for the masked indices
    inp_mask[:] = False
    inp_mask[mask_indices] = True
    print(inp_mask)

    # 4. Create labels for masked tokens
    labels = -1 * np.ones(encoded_texts.shape, dtype=int)
    labels[inp_mask] = encoded_texts[inp_mask]
    print(labels)

    # 5. Create masked input
    encoded_texts_masked = np.copy(encoded_texts)
    encoded_texts_masked[inp_mask] = config.mask_token_id  # mask token is the last in the dict
    print(encoded_texts_masked)

    # 6. Define loss function weights
    sample_weights = np.ones(labels.shape)
    sample_weights[labels == -1] = 0
    print(sample_weights)

    # 7. Finally define labels
    y_labels = np.copy(encoded_texts)

    return encoded_texts_masked, y_labels, sample_weights, boards



def test_encoding_2():
    move_sequence = ['a1a2', 'd5c4', 'c2c4', 'g7g6', 'b1c3', 'd7d5', 'c1f4', 'f8g7', 'f4e5', 'd5c4', 'e2e3', 'b8c6',
                     'd1a4', 'e8g8', 'e5f6', 'g7f6', 'f1c4', 'a7a6', 'c4d5', 'b7b5', 'a4d1', 'c8b7', 'a2a3', 'e7e6',
                     'd5f3', 'c6a5', 'f3b7', 'a5b7', 'b2b4']
    last_half = ['c7c5', 'b4c5', 'b7c5', 'g1f3', 'd8a5', 'd1c2', 'c5a4', 'a1c1', 'a8c8', 'e1g1', 'a5c3', 'c2e2', 'c3a3',
                 'c1c2', 'c8c2', 'e2c2', 'a3c3', 'c2e4', 'f8c8', 'g2g3', 'c3c2', 'e4b7', 'c2c6']
    boards = None

    move_string = ' '.join(move_sequence)
    encoded_texts = config.encode(move_string)
    print(move_string)
    print(encoded_texts)

    # 1. Find possible masking positions
    inp_mask = tf.random.uniform(encoded_texts.shape) < 10  # Initialize all to True
    inp_mask = tf.logical_and(inp_mask, encoded_texts > 2)  # Determine which tokens can be masked
    print(inp_mask)

    # 2. Find indices where n tokens can be masked in a row
    mask_length = 3
    # indices = tf.where(tf.logical_and(inp_mask[:-2], inp_mask[1:-1], inp_mask[2:]))[:, 0]

    indices = tf.where(
        tf.logical_and(
            tf.logical_and(inp_mask[:-2], inp_mask[1:-1]),
            inp_mask[2:]
        )
    )[:, 0]

    mask_start = tf.random.shuffle(indices)[0]  # Choose a random index to start masking
    mask_indices = tf.range(mask_start, mask_start + mask_length)
    print(mask_indices)

    # 3. Set all entries in inp_mask to False except for the masked indices
    print('Mask indices:', mask_indices.shape)
    print(type(mask_indices))
    print('inp_mask:', inp_mask.shape)
    inp_mask = tf.scatter_nd(tf.expand_dims(mask_indices, 1), tf.ones_like(mask_indices, dtype=tf.bool), inp_mask.shape)
    print(inp_mask)
    print(type(inp_mask))
    exit(0)

    # 4. Create labels for masked tokens
    labels = -1 * tf.ones(encoded_texts.shape, dtype=tf.int32)
    labels = tf.where(inp_mask, encoded_texts, labels)
    print(labels)

    # 5. Create masked input
    encoded_texts_masked = tf.where(inp_mask, config.mask_token_id * tf.ones_like(encoded_texts), encoded_texts)
    print(encoded_texts_masked)

    # 6. Define loss function weights
    sample_weights = tf.ones(labels.shape)
    sample_weights = tf.where(labels == -1, tf.zeros_like(sample_weights), sample_weights)
    print(sample_weights)

    # 7. Finally define labels
    y_labels = tf.identity(encoded_texts)
    print(y_labels)


def test_encoding_3():
    move_sequence = ['a1a2', 'd5c4', 'c2c4', 'g7g6', 'b1c3', 'd7d5', 'c1f4', 'f8g7', 'f4e5', 'd5c4', 'e2e3', 'b8c6',
                     'd1a4', 'e8g8', 'e5f6', 'g7f6', 'f1c4', 'a7a6', 'c4d5', 'b7b5', 'a4d1', 'c8b7', 'a2a3', 'e7e6',
                     'd5f3', 'c6a5', 'f3b7', 'a5b7', 'b2b4']
    last_half = ['c7c5', 'b4c5', 'b7c5', 'g1f3', 'd8a5', 'd1c2', 'c5a4', 'a1c1', 'a8c8', 'e1g1', 'a5c3', 'c2e2', 'c3a3',
                 'c1c2', 'c8c2', 'e2c2', 'a3c3', 'c2e4', 'f8c8', 'g2g3', 'c3c2', 'e4b7', 'c2c6']
    boards = None

    move_string = ' '.join(move_sequence)
    encoded_texts = config.encode(move_string)
    print(move_string)
    print(encoded_texts)

    # 1. Find possible masking positions
    inp_mask = tf.random.uniform(encoded_texts.shape) <= 1.0
    inp_mask = tf.logical_and(inp_mask, encoded_texts > 2)

    # 2. Find indices where n tokens can be masked in a row
    true_indices = tf.squeeze(tf.where(inp_mask), axis=1)
    true_indices = true_indices[:-2]
    rand_idx = tf.random.uniform(shape=[], maxval=tf.shape(true_indices)[0], dtype=tf.int32)
    mask_start = tf.gather(true_indices, rand_idx)
    mask_length = 3
    mask_indices = tf.range(mask_start, mask_start + mask_length)

    print(inp_mask)
    print(inp_mask.shape)
    print(mask_indices)
    print(true_indices.shape)
    exit(0)

    # 3. Set all entries in inp_mask to False except for the masked indices
    inp_mask = tf.scatter_nd(tf.expand_dims(mask_indices, 1), tf.ones_like(mask_indices, dtype=tf.bool),
                             inp_mask.shape)

    # 4. Create labels for masked tokens
    labels = -1 * tf.ones(encoded_texts.shape, dtype=tf.int64)
    labels = tf.where(inp_mask, encoded_texts, labels)

    # 5. Create masked input
    encoded_texts_masked = tf.identity(encoded_texts)
    mask_token_id = config.mask_token_id
    encoded_texts_masked = tf.where(inp_mask, mask_token_id * tf.ones_like(encoded_texts), encoded_texts)

    # 6. Define loss function weights
    sample_weights = tf.ones(labels.shape, dtype=tf.int64)
    sample_weights = tf.where(tf.equal(labels, -1), tf.zeros_like(labels), sample_weights)

    # 7. Finally define labels
    y_labels = tf.identity(encoded_texts)

    return encoded_texts_masked, y_labels, sample_weights, boards





def parse_epd_line(line):
    epd_parts = line.split(";")
    fen = epd_parts[0].strip()
    evaluation = float(epd_parts[1].split("=")[1].strip())
    return fen, evaluation

def process_epd_file(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()

    positions_and_evaluations = [parse_epd_line(line) for line in lines]

    for fen, evaluation in positions_and_evaluations:
        board = chess.Board(fen)

        # Perform any operation you want with the board and evaluation, e.g. print the board and evaluation
        print(board)
        print("Evaluation:", evaluation)



def test_epd():
    epd_directory = "path/to/your/epd_directory"

    # Iterate through all the .epd files in the directory
    for file_name in os.listdir(epd_directory):
        if file_name.endswith(".epd"):
            file_path = os.path.join(epd_directory, file_name)
            process_epd_file(file_path)



if __name__ == '__main__':
    test_engine()