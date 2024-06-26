from keras import layers
from hydra import config


class MoveMaskPrediction(layers.Layer):

    def __init__(self):
        super(MoveMaskPrediction, self).__init__()

        self.move_mask_prediction = layers.Dense(config.vocab_size, name="move_prediction_head", activation="softmax")

    def __call__(self, inputs):
        move_mask = self.move_mask_prediction(inputs)
        return move_mask


















