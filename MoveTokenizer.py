# --> Python Imports
import os
import numpy as np
import pickle
import itertools
from copy import deepcopy

# --> Tensorflow Imports
from tqdm import tqdm
from keras import layers
import tensorflow as tf
from keras_nlp.tokenizers import Tokenizer


# --> Threading Imports
from concurrent.futures import ThreadPoolExecutor

# --> Chess Imports
import chess
import chess.pgn as chess_pgn



class MoveTokenizer(Tokenizer):
    def __init__(self, vocab_file, mask=False, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable=trainable, name=name, dtype=dtype, dynamic=dynamic, **kwargs)
        with open(vocab_file, 'rb') as f:
            self.vocab = list(pickle.load(f))
            self.vocab.sort()
            print(type(self.vocab))
        self.num_move_tokens = len(self.vocab)
        if mask is True:
            self.vocab.insert(0, '[MASK]')
        self.id_to_token_map = {i: token for i, token in enumerate(self.vocab)}
        self.token_to_id_map = {token: i for i, token in enumerate(self.vocab)}

    def tokenize(self, inputs, output_sequence_length=None, *args, **kwargs):
        tokens = tf.strings.split(inputs, sep=" ")
        token_ids = tf.map_fn(lambda token: self.token_to_id(token.numpy().decode('utf-8')), tokens, dtype=tf.int32)

        if output_sequence_length:
            token_ids_shape = tf.shape(token_ids)
            pad_length = self.output_sequence_length - token_ids_shape[-1]
            paddings = tf.cond(pad_length > 0, lambda: [[0, 0], [0, pad_length]], lambda: [[0, 0], [0, 0]])
            token_ids = tf.pad(token_ids, paddings, "CONSTANT", constant_values=0)

        return token_ids

    def detokenize(self, inputs, *args, **kwargs):
        tokens = tf.map_fn(lambda id: self.id_to_token(id.numpy()), inputs, dtype=tf.string)
        detokenized_text = tf.strings.reduce_join(tokens, separator=" ", axis=-1)
        return detokenized_text

    def get_vocabulary(self):
        return self.vocab

    def vocabulary_size(self):
        return len(self.vocab)

    def token_to_id(self, token: str):
        return self.token_to_id_map.get(token, None)

    def id_to_token(self, id: int):
        return self.id_to_token_map.get(id, None)





if __name__ == '__main__':
    mt = MoveTokenizer('tokens/tokens_1946.pkl', mask=True)
    print(mt.vocab)
    print(mt.token_to_id('[MASK]'))
    print(mt.token_to_id('a1e1'))

    to_tokenize = 'a1e1 a1b3 a1g1'
    print(mt.tokenize(to_tokenize))