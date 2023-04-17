
import tensorflow as tf
import keras
from keras import layers
from keras_nlp.layers import MaskedLMMaskGenerator



# Subclass Keras MaskedLMMaskGenerator class to mask a tensor of shape (batch_size, 8, 8, 12)


class BoardMask(layers.Layer):



    def __init__(self,
        vocabulary_size,
        mask_selection_rate,
        mask_token_id,


        # mask_selection_length=None,
        board_mask_size=(3, 3, 3),





        unselectable_token_ids=[0],
        random_token_rate=0.1,
        **kwargs
     ):
        super(BoardMask, self).__init__(**kwargs)



