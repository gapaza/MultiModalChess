import os
import pickle


#######################
##### Directories #####
#######################
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
datasets_dir = os.path.join(root_dir, 'datasets')
models_dir = os.path.join(root_dir, 'models')



######################
##### Vocabulary #####
######################
vocab_file = os.path.join(root_dir, 'tokens', 'tokens_1946.pkl')
vocab = []
with open(vocab_file, 'rb') as f:
    vocab = list(pickle.load(f))
    vocab.sort()
vocab_size = len(vocab)


####################
##### Datasets #####
####################
human_games_file = os.path.join(root_dir, 'games', 'human-training-games.pgn')
human_positions_file = os.path.join(root_dir, 'positions', 'human-training-positions-72753.pkl')



#############################
##### Training Settings #####
#############################
train_dataset = 'train-dataset-72753'
val_dataset = 'val-dataset-72753'
model_name = 'hydrachess'
epochs = 10
batch_size = 32
seq_length = 128
# find vocab size by len of list in tokens file
embed_dim = 256
encoder_dense_dim = 2048
encoder_heads = 12