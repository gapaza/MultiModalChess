import chess
import numpy as np






class BoardOperations:

    @staticmethod
    def board_to_tensor(board):
        tensor = np.zeros((8, 8, 12))
        for square, piece in board.piece_map().items():
            rank, file = divmod(square, 8)
            piece_index = BoardOperations.piece_to_index(piece)
            tensor[7 - rank, file, piece_index] = 1
        return tensor

    @staticmethod
    def piece_to_index(piece):
        piece_order = ['P', 'N', 'B', 'R', 'Q', 'K']
        index = piece_order.index(piece.symbol().upper())
        # If the piece is black, add 6 to the index (to cover the range 0-11)
        if piece.color == chess.BLACK:
            index += 6
        return index

