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

# --> Mask position
mask_pos = 5




def get_board_tensor_from_moves(move_tokens, move_idx):
    move_idx = move_idx.numpy()
    moves = [config.id2token[token_id] for token_id in move_tokens.numpy()]
    board = chess.Board()
    try:
        for i in range(move_idx+1):
            board.push_uci(moves[i])
    except Exception as e:
        print('--> INVALID MOVE')
    return BoardOperations.board_to_tensor(board)



def custom_preprocess(moves):
    print('Custom Preprocess')
    encoded_texts = config.tokenizer(moves)

    # 1. Find possible masking positions
    # inp_mask.shape: (128,)
    inp_mask = tf.random.uniform(encoded_texts.shape) <= 1.0
    inp_mask = tf.logical_and(inp_mask, encoded_texts > 2)

    true_indices = tf.where(inp_mask)
    first_true_index = true_indices[0]
    inp_mask = tf.concat([
        inp_mask[:first_true_index[0]],
        [False],
        inp_mask[first_true_index[0] + 1:]
    ], axis=0)

    last_true_index = true_indices[-1]
    inp_mask = tf.concat([
        inp_mask[:last_true_index[0]],
        [False],
        inp_mask[last_true_index[0] + 1:]
    ], axis=0)

    # 2. Find center point where 3 tokens can be masked by a mask slide
    true_indices = tf.squeeze(tf.where(inp_mask), axis=1)
    rand_idx = tf.random.uniform(shape=[], maxval=tf.shape(true_indices)[0], dtype=tf.int32)
    mask_center = tf.gather(true_indices, rand_idx)
    mask_start = mask_center - 1
    mask_length = 3
    mask_indices = tf.range(mask_start, mask_start + mask_length)

    # 3.1 Get board tensor using tf.py_function to call get_board_tensor_from_moves
    board_tensor = tf.py_function(get_board_tensor_from_moves, [encoded_texts, mask_center], tf.int32)
    board_tensor.set_shape((8, 8, 12))

    # 3. Set all entries in inp_mask to False except for the masked indices
    inp_mask = tf.zeros((128,), dtype=tf.bool)
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

    # 8. Print all returns
    print('Encoded Texts Masked: ', encoded_texts_masked)
    print('Sample Weights: ', sample_weights)
    print('Y Labels: ', y_labels)
    print('Board Tensor: ', board_tensor)

    return encoded_texts_masked, y_labels, sample_weights, board_tensor






def test_preprocess(inputs, board, mask_pos):
    encoded_texts = inputs

    # 1. Find possible masking positions
    # inp_mask.shape: (128,)
    inp_mask = tf.random.uniform(encoded_texts.shape) <= 1.0
    inp_mask = tf.logical_and(inp_mask, encoded_texts > 2)

    # --> Mask of length 3
    mask_length = 3
    mask_start = mask_pos - 1
    mask_indices = tf.range(mask_start, mask_start + mask_length)

    # --> Mask of length 5
    # mask_length = 5
    # mask_start = mask_pos - 2
    # mask_indices = tf.range(mask_start, mask_start + mask_length)

    # 3. Set all entries in inp_mask to False except for the masked indices
    inp_mask = tf.zeros((128,), dtype=tf.bool)
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

    # 8. Print all returns
    print('Encoded Texts Masked: ', encoded_texts_masked)
    print('Sample Weights: ', sample_weights)
    print('Y Labels: ', y_labels)
    # print('Board: ', board)

    return encoded_texts_masked, y_labels, sample_weights, board










if __name__ == '__main__':
    custom_preprocess(moves_str)