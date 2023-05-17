import os
import pickle
import tensorflow as tf


#######################
##### Directories #####
#######################

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
datasets_dir = os.path.join(root_dir, 'datasets')
models_dir = os.path.join(root_dir, 'models')
positions_dir = os.path.join(root_dir, 'positions')
games_dir = os.path.join(root_dir, 'games')
chess_com_games_dir = os.path.join(games_dir, 'chess-com')
tokens_dir = os.path.join(root_dir, 'tokens')
checkpoints_dir = os.path.join(root_dir, 'checkpoints')
board_attention_dir = os.path.join(root_dir, 'hydra', 'board_attention')


################################
##### Stockfish Data Files #####
################################

stockfish_data_dir = os.path.join(root_dir, 'evaluations')


########################################
##### Parsing Games Into Positions #####
########################################

# --> Game File Input
# games_file = os.path.join(root_dir, 'games', 'chess-com-gm-games.pgn')
games_file = os.path.join(root_dir, 'games', 'human-training-games.pgn')
# games_file = os.path.join(root_dir, 'games', 'computer', 'ccrl-40-15-elo-3400.pgn')

# --> Game Directory Input
games_file_dir = os.path.join(root_dir, 'games', 'chess-com-gm-games')
# games_file_dir = os.path.join(root_dir, 'games', 'ccrl-40-15-elo-3400')
games_file_dir = os.path.join(root_dir, 'games', 'human-training-games')
# games_file_dir = os.path.join(root_dir, 'games', 'human-training-games-san')


# --> Eval Directory
eval_positions_dir = os.path.join(positions_dir, 'all-epds')





###########################################
##### Parsing Positions Into Datasets #####
###########################################

# positions_load_dir = os.path.join(positions_dir, 'chess-com-gm-games')
# positions_load_dir = os.path.join(positions_dir, 'human-training-games')
# positions_load_dir = os.path.join(positions_dir, 'ccrl-40-15-elo-3400-095247')
positions_load_dir = os.path.join(positions_dir, 'human-training-games-141727')



#######################
##### Fine-Tuning #####
#######################

fine_tuning_evaluations_dir = os.path.join(root_dir, 'games', 'fine-tuning')




#############################
##### Training Settings #####
#############################
save_dataset = False
# train_dataset = 'human-training-games-141727-train-120410'
# val_dataset = 'human-training-games-141727-val-120410'
train_dataset = 'human-training-games-141727-train-combined-64b-175839'
val_dataset = 'human-training-games-141727-val-combined-64b-175839'
model_name = 'hydrachess'
epochs = 3
batch_size = 32  # 32 64 128 256 512 1024
seq_length = 128  # 256 max
# find vocab size by len of list in tokens file
embed_dim = 256  # 512 too much
encoder_dense_dim = 1024  # 2048
encoder_heads = 48
num_sparse_board = 3


vt_dense_dim = 1024
vt_img_size = 8
vt_patch_size = 1
vt_num_patches = (vt_img_size // vt_patch_size) ** 2
vt_epsilon = 1e-6
vt_heads = 48




##############################
### Tokenizer + Vocabulary ###
##############################
import tensorflow as tf
from keras.layers import TextVectorization
import re


def custom_standardization(input_data):
    # lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(input_data, "<br />", " ")
    return tf.strings.regex_replace(
        stripped_html, "[%s]" % re.escape("!#$%&'()*+,-./:;<=>?@\^_`{|}~"), ""
    )


special_tokens = ["[pos]", "[mask]"]
num_special_tokens = len(special_tokens) + 2
vocab_file = os.path.join(root_dir, 'tokens', 'tokens_1969_merged.pkl')  # tokens_1966.pkl, tokens_1968_chesscom
vocab = []
with open(vocab_file, 'rb') as f:
    vocab = list(pickle.load(f))
    # remove empty string and [UNK]
    if '' in vocab:
        vocab.remove('')
    vocab.sort()
vocab = special_tokens + vocab
vocab_size = len(vocab)
tokenizer = TextVectorization(
    max_tokens=vocab_size + 2,
    output_mode="int",
    standardize=custom_standardization,
    output_sequence_length=seq_length,
)
tokenizer.set_vocabulary(vocab)
vocab = tokenizer.get_vocabulary()
vocab_size = len(vocab)
mask_token_id = tokenizer(["[mask]"]).numpy()[0][0]
padding_token_id = tokenizer(['']).numpy()[0][0]
pos_token_id = tokenizer(["[pos]"]).numpy()[0][0]
id2token = dict(enumerate(tokenizer.get_vocabulary()))
token2id = {y: x for x, y in id2token.items()}

def encode(input):
    encoded_input = tokenizer(input)
    return encoded_input.numpy()

def encode_tf(input):
    encoded_input = tokenizer(tf.expand_dims(input, axis=0))
    encoded_input = tf.squeeze(encoded_input, axis=0)
    return encoded_input

def encode_tf_batch(input):
    encoded_input = tokenizer(input)
    return encoded_input


print('--> FINISHED: config.py')

# Commands
# scp -i ~/keys/gabe-master.pem ./human-training-games-299k.zip ubuntu@3.17.77.24:/home/ubuntu/MultiModalChess/datasets
# scp -i ~/keys/gabe-master.pem ubuntu@18.221.115.53:/home/ubuntu/MultiModalChess/positions/human-training-games-141727.zip .
# scp -i ~/keys/gabe-master.pem ./human-training-games-141727.zip ubuntu@18.221.115.53:/home/ubuntu/MultiModalChess/positions


