import os


root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

seq_length = 128

# find vocab size by len of list in tokens file
vocab_size = 1946
encoder_dense_dim = 2048
encoder_heads = 12