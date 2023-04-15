from keras import layers


class BoardEmbedding(layers.Layer):

    def __init__(self):
        super(BoardEmbedding, self).__init__()

        self.board_conv2d_1 = layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu")
        self.board_conv2d_2 = layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding='same')
        self.board_flatten = layers.Flatten()
        self.board_dense = layers.Dense(256, activation="relu")
        self.board_reshape = layers.Reshape((1, -1), name='board_embedding')


    def __call__(self, inputs):
        board_embedding = self.board_conv2d_1(inputs)
        board_embedding = self.board_conv2d_2(board_embedding)
        board_embedding = self.board_flatten(board_embedding)
        board_embedding = self.board_dense(board_embedding)
        board_embedding = self.board_reshape(board_embedding)
        return board_embedding




