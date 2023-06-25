import tensorflow as tf
import tensorflow_ranking as tfr
from hydra import config
from preprocess.strategies.tf_utils import get_move_masking_positions, \
        constrain_move_mask_window_positions, generate_random_mask_window, \
        pad_existing_sequence_moves, apply_move_mask, \
        generate_random_mask_window_long, get_move_masking_positions_batch, constrain_move_mask_window_positions_batch
from preprocess.strategies.py_utils import get_sequence_board_tensor, get_board_tensor_at_move




def move_ranking_batch(norm_scores, uci_moves, prev_moves):
    # (None, 1973) (None, 1973) (None, 128)
    # norm_scores, uci_moves, prev_moves = all_inputs
    output_batch = tf.map_fn(
            move_ranking,  # The function to apply to each element in the batch
            (norm_scores, uci_moves, prev_moves),  # The input tensor with shape (None, 128)
            fn_output_signature = (
                tf.TensorSpec(shape=(128,), dtype=tf.int64),                    # current_position
                tf.TensorSpec(shape=(config.vocab_size,), dtype=tf.int64),      # ranked move labels
                tf.TensorSpec(shape=(config.vocab_size,), dtype=tf.float32),    # ranked move relevancy scores
                tf.TensorSpec(shape=(8, 8, 12), dtype=tf.int64),                # board_tensor
            )
            # The expected output shape and data type
    )
    return output_batch


@tf.function
def move_ranking(all_inputs):
    candidate_scores, candidate_moves, previous_moves = all_inputs
    # (1973,) (1973,) (128,)


    previous_moves_encoded = config.encode_tf(previous_moves)
    candidate_moves_encoded = config.encode_tf_long(candidate_moves)

    board_tensor = tf.py_function(get_sequence_board_tensor, [previous_moves_encoded], tf.int64)
    board_tensor.set_shape((8, 8, 12))

    return previous_moves_encoded, candidate_moves_encoded, candidate_scores, board_tensor










#
# @tf.function
# def test_move_ranking(first_element):
#     # your test code here
#     prev_moves_encoded, uci_moves_encoded, norm_scores, board_tensor = move_ranking(first_element)

@tf.function
def loss_fn(y_true, y_pred):
    print('y_true:', y_true)
    print('y_pred:', y_pred)
    results = tfr.losses._approx_ndcg_loss(y_true, y_pred)
    return results





if __name__ == '__main__':
    print('Testing Move Ranking')

    # 1. Test Position
    input_obj = {
        'norm_scores': [1.0, 0.85, 0.84],
        'uci_moves': ['b8c6', 'a7a6', 'g7g6'],
        'prev_moves': ['g1f3', 'c7c5', 'e2e4', 'd7d6', 'd2d4', 'c5d4', 'f3d4', 'g8f6', 'b1c3']
    }
    positions = [input_obj] * 99


    # 2. Create Dataset
    import preprocess.FineTuningDatasetGenerator as pp
    dataset = pp.FineTuningDatasetGenerator.create_and_pad_dataset(positions)
    dataset = dataset.batch(config.batch_size, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(move_ranking_batch, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    # 3. Parse Dataset
    # first_element = next(iter(dataset.take(1)))

    # 4. Test Move Ranking
    # prev_moves_encoded, uci_moves_encoded, norm_scores, board_tensor = first_element


    # Loss function
    y_true = [[1., 0.]]
    y_pred = [[0.6, 0.8]]

    # transform y_true and y_pred to tensors
    y_true = tf.convert_to_tensor(y_true)
    y_pred = tf.convert_to_tensor(y_pred)

    results = loss_fn(y_true, y_pred)






