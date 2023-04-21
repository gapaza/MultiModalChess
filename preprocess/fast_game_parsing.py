import os
from hydra import config
import chess.pgn
import concurrent.futures
from tqdm import tqdm


def split_pgn_file(large_pgn_file, max_games_per_file=100000):
    large_pgn_file_name = os.path.basename(large_pgn_file).split(".")[0]
    output_dir = os.path.join(config.games_dir, large_pgn_file_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    game_count = 0
    file_count = 0
    buffer = []
    progress_bar = tqdm(desc="Splitting PGN", unit="Games")

    with open(large_pgn_file, 'r', errors="ignore") as pgn_file:
        curr_newline_count = 0
        for line in pgn_file:
            buffer.append(line)
            if line.strip() == '':
                curr_newline_count += 1
                if curr_newline_count == 2:
                    game_count += 1
                    progress_bar.update(1)
                    if game_count == max_games_per_file:
                        file_name = os.path.join(output_dir, f"pgn_chunk_{file_count}_{game_count}.pgn")
                        with open(file_name, 'w') as chunk_file:
                            chunk_file.writelines(buffer)
                        buffer = []
                        game_count = 0
                        file_count += 1
                    curr_newline_count = 0
    if buffer:
        file_name = os.path.join(output_dir, f"pgn_chunk_{file_count}_{game_count}.pgn")
        with open(file_name, 'w') as chunk_file:
            chunk_file.writelines(buffer)



def test_saved_file():
    file_path = os.path.join(config.games_dir, "human-training-games", "pgn_chunk_7_100000.pgn")
    with open(file_path, 'r', errors="ignore") as pgn_file:
        game = chess.pgn.read_game(pgn_file)
        print(game.headers)
        print(game.mainline_moves())
        print(game.headers['Result'])


if __name__ == '__main__':
    test_saved_file()
    # game_batch = split_pgn_file(config.games_file, 100000)


