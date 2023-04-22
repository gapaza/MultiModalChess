import chess
import chess.pgn as chess_pgn
from hydra import config
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import concurrent.futures
import os
import chess.engine
from chess.engine import PovScore
import pickle

def test_engine():
    import chess.engine
    stockfish_path = '/opt/homebrew/Cellar/stockfish/15.1/bin/stockfish'
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    print(engine.id)

    # Set the maximum number of games to process
    max_games = 1

    # Open the PGN file
    with open(config.games_file) as pgn_file:
        progress_bar = tqdm(range(max_games), desc="Processing games", unit="game")
        for _ in progress_bar:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            game_moves = list(move for move in game.mainline_moves())
            game_moves_uci = list(move.uci() for move in game_moves)
            middle_game = len(game_moves) // 2
            board = game.board()
            next_move = game_moves[0]
            for x in range(middle_game):
                move = game_moves[x]
                next_move = game_moves[x + 1]
                board.push(move)
            print('Analyzing board')
            info = engine.analyse(board, chess.engine.Limit(time=5), multipv=5)
            for idx, variation in enumerate(info, start=1):
                move = variation.get("pv")[0]
                score = variation.get("score")
                if board.turn == chess.BLACK:
                    score = score.relative.score()
                print(f"{idx}. {move} ({score}) {middle_game}")


# This function takes a list of strings, where each is a whitespace separated list of uci moves representing a full game.
# It returns a list of dicts, where each dict contains the game's moves, and a list of dicts, where each dict corresponds to the game's current position and has the top 5 stockfish recommended uci moves and their evaluation
# For each game position, stockfish is evaluated to depth 20.
import chess
import chess.engine
def generate_stockfish_data_fast(games):
    stockfish_path = '/opt/homebrew/Cellar/stockfish/15.1/bin/stockfish'

    def analyze_position(board, depth=20, multipv=5):
        with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
            info = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=multipv)
            top_5_moves = []
            for idx, variation in enumerate(info, start=1):
                top_move = variation.get("pv")[0]
                score = variation.get("score")
                if board.turn == chess.BLACK:
                    score = score.black().score()
                elif board.turn == chess.WHITE:
                    score = score.white().score()
                top_5_moves.append({
                    'move': top_move.uci(),
                    'evaluation': score
                })
        return top_5_moves

    def process_game_chunk(game_chunk):
        chunk_evaluation_data = []
        for game in game_chunk:
            moves = list(move.uci() for move in game.mainline_moves())
            position_data = []
            board = chess.Board()

            for idx1, move in enumerate(moves):
                print('Analyzing board: ', board.move_stack)
                top_5_moves = analyze_position(board.copy())
                position_data.append({
                    'position': idx1,
                    'top_5_moves': top_5_moves
                })
                board.push_uci(move)

            chunk_evaluation_data.append({
                'game_moves': moves,
                'position_data': position_data
            })

        return chunk_evaluation_data

    def chunk_games(games, n):
        return [games[i:i + n] for i in range(0, len(games), n)]

    num_workers = 24
    chunk_size = len(games) // num_workers

    evaluation_data = []
    game_chunks = chunk_games(games, chunk_size)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_game_chunk, game_chunk) for game_chunk in game_chunks]
        for future in concurrent.futures.as_completed(futures):
            evaluation_data.extend(future.result())

    # pickle evaluation_data with config.stockfish_data_dir
    with open(os.path.join(config.stockfish_data_dir, 'stockfish_data_'+str(len(evaluation_data))+'.pkl'), 'wb') as f:
        pickle.dump(evaluation_data, f)
    return evaluation_data












def test_data():
    test_file = '/Users/gapaza/repos/gabe/MultiModalChess/evaluations/stockfish_data_1.pkl'
    # unpickle file
    with open(test_file, 'rb') as f:
        data = pickle.load(f)
    for game in data:
        moves = game['game_moves']
        position_data = game['position_data']
        for idx, move in enumerate(moves):
            move_data = position_data[idx]
            print(move_data)
            print(move)

if __name__ == '__main__':

    # open pgn file
    pgn_file = open(config.games_file)

    # get 30 games
    games = []
    for i in range(24):
        game = chess.pgn.read_game(pgn_file)
        games.append(game)

    stockfish_data = generate_stockfish_data_fast(games)
    print(stockfish_data)


# scp -i ~/keys/gabe-master.pem ./human-training-games-3448k.zip ubuntu@3.142.199.188:/home/ubuntu/MultiModalChess/datasets