import os
import pickle


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
games_file = os.path.join(root_dir, 'games', 'chess-com-gm-games.pgn')
# games_file = os.path.join(root_dir, 'games', 'computer', 'ccrl-40-15-elo-3400.pgn')

# --> Game Directory Input
games_file_dir = os.path.join(root_dir, 'games', 'chess-com-gm-games')




eval_positions_dir = os.path.join(positions_dir, 'all-epds')





###########################################
##### Parsing Positions Into Datasets #####
###########################################

# positions_load_dir = os.path.join(positions_dir, 'chess-com-gm-games')
positions_load_dir = os.path.join(positions_dir, 'human-training-games')




#############################
##### Training Settings #####
#############################
train_dataset = 'human-training-games-training-299k'
val_dataset = 'human-training-games-validation-299k'
model_name = 'hydrachess'
epochs = 30
batch_size = 64  # 32 64 128
seq_length = 128  # 256 max
# find vocab size by len of list in tokens file
embed_dim = 64  # 512 too much
encoder_dense_dim = 2048  # 2048
encoder_heads = 48
num_sparse_board = 3
visual_transformer_layers = 4
visual_transformer_heads = 12
visual_transformer_units = [
    embed_dim * 2,
    embed_dim,
]
vanilla_viz_transformer = False





##############################
### Tokenizer + Vocabulary ###
##############################
import tensorflow as tf
from keras.layers import TextVectorization
import re


def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")
    return tf.strings.regex_replace(
        stripped_html, "[%s]" % re.escape("!#$%&'()*+,-./:;<=>?@\^_`{|}~"), ""
    )


special_tokens = ["[pos]", "[mask]"]
num_special_tokens = len(special_tokens) + 2
vocab_file = os.path.join(root_dir, 'tokens', 'tokens_1966.pkl')  # tokens_1966.pkl, tokens_1968_chesscom
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


print('--> FINISHED: config.py')

# Commands
# scp -i ~/keys/gabe-master.pem ./human-training-games-299k.zip ubuntu@3.17.77.24:/home/ubuntu/MultiModalChess/datasets


