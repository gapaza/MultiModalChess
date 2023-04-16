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
# human_games_file = os.path.join(root_dir, 'games', 'human-training-games.pgn')
human_games_file = os.path.join(root_dir, 'games', 'computer-training-games.pgn')
# human_positions_file = os.path.join(root_dir, 'positions', 'human-training-positions-743847.pkl')
human_positions_file = os.path.join(root_dir, 'positions', 'human-training-positions-72753.pkl')
# human_positions_file = os.path.join(root_dir, 'positions', 'computer-training-positions-1373003.pkl')




#############################
##### Training Settings #####
#############################
train_dataset = 'train-dataset-1373003'  # 72753 1373003
val_dataset = 'val-dataset-1373003'
model_name = 'hydrachess'
epochs = 10
batch_size = 64  # 32 64 128
seq_length = 128  # 256 max
# find vocab size by len of list in tokens file
embed_dim = 256  # 512 too much
encoder_dense_dim = 2048  # 2048
encoder_heads = 12