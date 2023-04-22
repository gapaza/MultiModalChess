import os
from hydra import config
import chess.pgn
import concurrent.futures
from tqdm import tqdm


def split_pgn_file(large_pgn_file, max_games_per_file=100000):
    large_pgn_file_name = os.path.basename(large_pgn_file).split(".")[0]
    large_pgn_file_name += "-san"
    output_dir = os.path.join(config.games_dir, large_pgn_file_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    game_count = 0
    file_count = 0
    buffer = []
    progress_bar = tqdm(desc="Splitting PGN", unit="Games")
    new_game = True
    with open(large_pgn_file, 'r', errors="ignore") as pgn_file:
        curr_newline_count = 0
        for line in pgn_file:
            if new_game and line.strip() == '':
                continue
            new_game = False
            buffer.append(line)
            if line.strip() == '':
                curr_newline_count += 1
                if curr_newline_count > 1:
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
                    new_game = True
    if buffer:
        file_name = os.path.join(output_dir, f"pgn_chunk_{file_count}_{game_count}.pgn")
        with open(file_name, 'w') as chunk_file:
            chunk_file.writelines(buffer)




def parse_chess_com_games():
    game_files = os.listdir(config.chess_com_games_dir)
    game_files.sort()
    total_lines = []

    # Iterate over PGN files and read their content
    for idx, game_file in enumerate(game_files):
        pgn_file_path = os.path.join(config.chess_com_games_dir, game_file)
        with open(pgn_file_path, 'r') as f:
            file_lines = f.readlines()
            total_lines.extend(file_lines)

    # Save the combined content to a new file
    combined_file_path = os.path.join(config.games_dir, 'chess-com-gm-games.pgn')
    with open(combined_file_path, 'w') as f:
        f.writelines(total_lines)




def test_saved_file():
    file_path = os.path.join(config.chess_com_games_dir, 'hikaru.pgn')
    bad_games = 0
    with open(file_path, 'r', errors="ignore") as pgn_file:

        game = chess.pgn.read_game(pgn_file)
        while game:
            try:
                game = chess.pgn.read_game(pgn_file)
            except:
                print("bad game")
                bad_games += 1
                continue
    print(bad_games)

if __name__ == '__main__':
    # test_saved_file()
    # hikaru_path = os.path.join(config.chess_com_games_dir, 'hikaru.pgn')
    split_pgn_file(config.games_file, 100000)
    # parse_chess_com_games()


