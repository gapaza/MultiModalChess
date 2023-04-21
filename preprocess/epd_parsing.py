import os
from hydra import config
import chess








def parse_epd_dir():
    print(config.eval_positions_dir)
    epd_files = os.listdir(config.eval_positions_dir)
    for epd_file in epd_files:
        epd_file_path = os.path.join(config.eval_positions_dir, epd_file)
        process_epd_file(epd_file_path)


def process_epd_file(file_path):
    print('Processing', file_path)
    with open(file_path, "r") as file:
        lines = file.readlines()
    positions_and_evaluations = [parse_epd_line(line) for line in lines]
    for fen, evaluation in positions_and_evaluations:
        board = chess.Board(fen)

def parse_epd_line(line):
    epd_parts = line.split(";")
    fen = epd_parts[0].strip()
    evaluation = float(epd_parts[1].split("=")[1].strip())
    return fen, evaluation





if __name__ == '__main__':
    parse_epd_dir()