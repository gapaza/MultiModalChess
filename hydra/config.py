import os
import pickle


#######################
##### Directories #####
#######################
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
datasets_dir = os.path.join(root_dir, 'datasets')
models_dir = os.path.join(root_dir, 'models')


####################
##### Datasets #####
####################
games_file = os.path.join(root_dir, 'games', 'human-training-games.pgn')
# games_file = os.path.join(root_dir, 'games', 'computer-training-games.pgn')

# positions_file = os.path.join(root_dir, 'positions', 'human-training-positions-627.pkl')
positions_file = os.path.join(root_dir, 'positions', 'human-training-positions-6224.pkl')
# positions_file = os.path.join(root_dir, 'positions', 'human-training-positions-72753.pkl')
# positions_file = os.path.join(root_dir, 'positions', 'human-training-positions-743847.pkl')
# positions_file = os.path.join(root_dir, 'positions', 'computer-training-positions-1373003.pkl')




#############################
##### Training Settings #####
#############################
train_dataset = 'train-dataset-627'
val_dataset = 'val-dataset-627'
model_name = 'hydrachess'
epochs = 30
batch_size = 64  # 32 64 128
seq_length = 128  # 256 max
# find vocab size by len of list in tokens file
embed_dim = 256  # 512 too much
encoder_dense_dim = 2048  # 2048
encoder_heads = 12






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


special_tokens=["[mask]"]
vocab_file = os.path.join(root_dir, 'tokens', 'tokens_1946.pkl')
vocab = []
with open(vocab_file, 'rb') as f:
    vocab = list(pickle.load(f))
    vocab.sort()
vocab_size = len(vocab)
tokenizer = TextVectorization(
    max_tokens=vocab_size + 2,
    output_mode="int",
    standardize=custom_standardization,
    output_sequence_length=seq_length,
)
tokenizer.set_vocabulary(vocab)
vocab = tokenizer.get_vocabulary()
vocab = vocab[2: vocab_size - len(special_tokens)] + ["[mask]"]
tokenizer.set_vocabulary(vocab)
vocab_size = len(vocab)
mask_token_id = tokenizer(["[mask]"]).numpy()[0][0]


id2token = dict(enumerate(tokenizer.get_vocabulary()))
token2id = {y: x for x, y in id2token.items()}


print('--> FINISHED: config.py')