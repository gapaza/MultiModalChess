import tensorflow as tf
import os
import time
from hydra import config
import re
import ast
import pandas as pd
import pickle
import json
import csv
from tqdm import tqdm
from preprocess.strategies.window_masking import rand_window, rand_window_batch

from preprocess.strategies.move_ranking import move_ranking, move_ranking_batch

class FineTuningDatasetGenerator:

    def __init__(self):

        # 1. Get positions directory
        self.data_file = os.path.join(config.stockfish_data_dir, 'bulk_positions.txt')
        self.intermediate_file = os.path.join(config.stockfish_data_dir, 'bulk_positions.pkl')
        self.time_stamp = time.strftime("%H%M%S")

        # 2. Get move files
        self.position_evals = self.parse_evaluation_file()

        # 3. Split files with 90% train and 10% validation
        split_idx = int(len(self.position_evals) * 0.9)
        self.train_positions, self.val_positions = self.position_evals[:split_idx], self.position_evals[split_idx:]

        print('Train Positions:', len(self.train_positions))
        self.train_dataset = self.parse_dataset(self.train_positions)
        print('Val Positions:', len(self.val_positions))
        self.val_dataset = self.parse_dataset(self.val_positions)



    def parse_evaluation_file(self):
        if os.path.exists(self.intermediate_file):
            print('Loading:', self.intermediate_file)
            with open(self.intermediate_file, 'rb') as f:
                eval_data = pickle.load(f)
                print('First Entry:', eval_data[8])
                return eval_data


        eval_data = []
        idx = 0
        with open(self.data_file, 'r') as f:
            file_lines = f.readlines()
            # Iterate over entries 1 through the end

            for idx, line in enumerate(file_lines[1:100]):
                matches = re.findall(r'\[.*?\]', line)
                if len(matches) != 2:
                    print('Error parsing line:', line)
                    continue
                moves = json.loads(matches[0])
                prev_moves = ast.literal_eval(matches[1])
                try:
                    cp_scores = []
                    for move in moves:
                        eval_str = move['eval']
                        if 'Cp(' in eval_str:
                            cp_score = int(re.search(r"Cp\((.+?)\)", eval_str).group(1))
                        elif 'Mate(' in eval_str:
                            mate_score = int(re.search(r"Mate\((.+?)\)", eval_str).group(1))
                            # Assign a large score for checkmate evaluations.
                            cp_score = 10000 if mate_score > 0 else -10000
                        cp_scores.append(cp_score)
                except Exception as e:
                    print('Error parsing line:', line, e)
                    continue
                if 'WHITE' in moves[0]['eval']:
                    best_score = max(cp_scores)
                else:
                    best_score = min(cp_scores)
                # Compute absolute differences and normalize.
                norm_scores = [abs(score - best_score) / 100 for score in cp_scores]

                # Invert the scores so that a higher score is better.
                norm_scores_inv = [round(1 - score, 3) for score in norm_scores]

                # sort moves wand norm_score together on norm_score with zip
                # then unzip them into two lists
                moves, norm_scores_inv_sorted = zip(*sorted(zip(moves, norm_scores_inv), key=lambda x: x[1], reverse=True))
                uci_moves = [move['move'] for move in moves]

                # --> Outputs:
                # 1. norm_scores: list of normalized evaluation scores for current position
                # 2. uci_moves: list of candidate uci moves for current position
                # 3. prev_moves: list of previous uci moves leading up to current position
                eval_data.append({
                    'norm_scores': norm_scores_inv_sorted,
                    'uci_moves': uci_moves,
                    'prev_moves': prev_moves,
                })
                idx += 1



        # intermediate_file = os.path.join(config.stockfish_data_dir, 'bulk_positions_'+str(idx)+'.pkl')
        with open(self.intermediate_file, 'wb') as f:
            pickle.dump(eval_data, f)

        return eval_data

    @staticmethod
    def create_and_pad_dataset(positions):
        all_norm_scores = []
        all_uci_moves = []
        all_prev_moves = []
        for position in positions:

            # 1. Pad norm_scores
            norm_scores = list(position['norm_scores'])
            while len(norm_scores) < config.vocab_size:
                norm_scores.append(0)
            all_norm_scores.append(norm_scores)

            # 2. Pad uci_moves
            uci_moves = position['uci_moves']
            while len(uci_moves) < config.vocab_size:
                uci_moves.append('')
            all_uci_moves.append(uci_moves)

            # 3. Pad prev_moves
            prev_moves = position['prev_moves']
            while len(prev_moves) < config.seq_length:
                prev_moves.append('')
            all_prev_moves.append(prev_moves)
        return tf.data.Dataset.from_tensor_slices((all_norm_scores, all_uci_moves, all_prev_moves))


    def parse_dataset(self, positions):

        #####################
        ### Preprocessing ###
        #####################

        dataset = self.create_and_pad_dataset(positions)
        dataset = dataset.batch(config.batch_size, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(move_ranking_batch, num_parallel_calls=tf.data.AUTOTUNE)
        # dataset = dataset.shuffle(3125)
        return dataset.prefetch(tf.data.AUTOTUNE)


















if __name__ == '__main__':
    dg = FineTuningDatasetGenerator()
