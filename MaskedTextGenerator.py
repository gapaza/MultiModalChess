import tensorflow as tf
import keras
from keras import layers
from keras.layers import TextVectorization
from dataclasses import dataclass
import pandas as pd
import numpy as np
import glob
import re
from pprint import pprint



class MaskedTextGenerator(keras.callbacks.Callback):
    def __init__(self, sample_tokens, top_k=5, id2token=None, mask_token_id=None):
        self.sample_tokens = sample_tokens
        self.mask_token_id = mask_token_id
        self.id2token = id2token
        self.k = top_k

    def decode(self, tokens):
        return " ".join([self.id2token[t] for t in tokens if t != 0])

    def convert_ids_to_tokens(self, id):
        return self.id2token[id]

    def on_epoch_end(self, epoch, logs=None):
        prediction = self.model.predict(self.sample_tokens)

        masked_index = np.where(self.sample_tokens == self.mask_token_id)
        masked_index = masked_index[1]
        mask_prediction = prediction[0][masked_index]

        top_indices = mask_prediction[0].argsort()[-self.k :][::-1]
        values = mask_prediction[0][top_indices]

        for i in range(len(top_indices)):
            p = top_indices[i]
            v = values[i]
            tokens = np.copy(self.sample_tokens[0])
            tokens[masked_index[0]] = p
            result = {
                "input_text": self.decode(self.sample_tokens[0].numpy()),
                "prediction": self.decode(tokens),
                "probability": v,
                "predicted mask token": self.convert_ids_to_tokens(p),
            }
            pprint(result)