from keras import layers
from hydra import config


class MovePrediction(layers.Layer):

    def __init__(self):
        super(MovePrediction, self).__init__()

        self.next_move_avg = layers.GlobalAveragePooling1D()
        self.next_move_dropout = layers.Dropout(0.5)
        self.next_move_prediction = layers.Dense(config.vocab_size, activation="softmax", name='next_move_prediction')



    def __call__(self, inputs):
        next_move = self.next_move_avg(inputs)
        next_move = self.next_move_dropout(next_move)
        next_move = self.next_move_prediction(next_move)
        return next_move

