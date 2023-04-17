import os
from hydra.config import root_dir
from keras.models import load_model
import chess
import random




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
        return random.choice(legal_moves)

    def random_move(self):
        legal_moves = list(self.board.legal_moves)
        return random.choice(legal_moves)

    def play_interactive_game(self):
        while not self.board.is_game_over():
            print(self.board)
            if self.board.turn == chess.WHITE:
                user_move = input("Your move (in UCI format, e.g. 'e2e4'): ")
                try:
                    move = chess.Move.from_uci(user_move)
                    if move in self.board.legal_moves:
                        self.board.push(move)
                    else:
                        print("Illegal move. Please enter a valid move.")
                except ValueError:
                    print("Invalid input. Please enter a valid move in UCI format.")
            else:
                cpu_move = self.random_move()
                print(f"CPU move: {cpu_move}")
                self.board.push(cpu_move)

        print("Game Over.")
        print(self.board.result())


if __name__ == "__main__":
    model_wrapper = ChessModelWrapper('hydrachess')
    model_wrapper.play_interactive_game()