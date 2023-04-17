from keras import layers
import keras
from hydra import config


class BoardEmbedding(layers.Layer):

    def __init__(self):
        super(BoardEmbedding, self).__init__()

        self.board_embedding = keras.Sequential([
            layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding='same'),
            layers.Dropout(0.5),
            layers.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding='same'),
            layers.Dropout(0.5),
            layers.Conv2D(filters=256, kernel_size=(3, 3), activation="relu", padding='same'),
            layers.Dropout(0.5),

            # layers.Reshape((64, 256), name='board_embedding'),

            # layers.Flatten(),
            # layers.Dense(8192, activation="relu"),
            # layers.Reshape((32, -1), name='board_embedding')




            layers.Flatten(),
            # layers.Dense(4096, activation="relu"),
            # layers.Dropout(0.5),
            layers.Dense(config.embed_dim, activation="relu"),
            layers.Reshape((1, -1), name='board_embedding')
        ])


    def __call__(self, inputs):
        board_embedding = self.board_embedding(inputs)
        return board_embedding




