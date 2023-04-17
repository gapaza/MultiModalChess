import os
from hydra.config import root_dir
from keras.models import load_model
from hydra import config
import chess
import random
import numpy as np

import tensorflow as tf

from inference.MaskedMoveGenerator import MaskedMoveGenerator



class ChessModelWrapper:
    def __init__(self, model_name):
        self.root_dir = root_dir
        self.model_dir = os.path.join(self.root_dir, 'models', model_name)
        self.model = load_model(self.model_dir)

        # --> Chess Board
        self.board = chess.Board()
        self.move_history = []










    def model_move(self):
        legal_moves = list(self.board.legal_moves)


        board_tensor = self.board_to_tensor(self.board)
        board_tensor = tf.expand_dims(board_tensor, axis=0)

        next_move = self.move_history + ['[mask]']
        all_moves = ' '.join(next_move)
        encoded_moves = self.encode(all_moves)
        encoded_moves = tf.expand_dims(encoded_moves, axis=0)

        print('Board Tensor: ', board_tensor.shape)
        print('All Moves: ', all_moves)
        print('All Moves Encoded: ', encoded_moves)






        # --> Get Move Probabilities
        predictions = self.model([board_tensor, encoded_moves], training=False)


        # --> Masked Move Generator
        self.masked_move_generator = MaskedMoveGenerator(all_moves, top_k=5)

        decoded = self.masked_move_generator.decode(encoded_moves)
        print('Decoded: ', decoded)







        return random.choice(legal_moves)



    def board_to_tensor(self, board):
        tensor = np.zeros((8, 8, 12))
        for square, piece in board.piece_map().items():
            rank, file = divmod(square, 8)
            piece_index = self.piece_to_index(piece)
            tensor[7 - rank, file, piece_index] = 1
        return tensor

    def piece_to_index(self, piece):
        piece_order = ['P', 'N', 'B', 'R', 'Q', 'K']
        index = piece_order.index(piece.symbol().upper())
        # If the piece is black, add 6 to the index (to cover the range 0-11)
        if piece.color == chess.BLACK:
            index += 6
        return index

    def encode(self, texts):
        encoded_texts = config.tokenizer(texts)
        return encoded_texts.numpy()





    def random_move(self):
        legal_moves = list(self.board.legal_moves)
        return random.choice(legal_moves)

    def play_interactive_game(self):
        while not self.board.is_game_over():
            print(self.board)
            print(self.move_history)
            if self.board.turn == chess.WHITE:
                user_move = input("Your move (in UCI format, e.g. 'e2e4'): ")
                try:
                    move = chess.Move.from_uci(user_move)
                    self.move_history.append(user_move)
                    if move in self.board.legal_moves:
                        self.board.push(move)
                    else:
                        print("Illegal move. Please enter a valid move.")
                except ValueError:
                    print("Invalid input. Please enter a valid move in UCI format.")
            else:
                cpu_move = self.random_move()
                self.model_move()
                self.move_history.append(cpu_move.uci())
                print(f"CPU move: {cpu_move}")
                self.board.push(cpu_move)

        print("Game Over.")
        print(self.board.result())


if __name__ == "__main__":
    model_wrapper = ChessModelWrapper('hydrachess')
    model_wrapper.play_interactive_game()