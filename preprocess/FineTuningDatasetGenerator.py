import tensorflow as tf
import os
import time
from hydra import config
from tqdm import tqdm
from preprocess.strategies.window_masking import rand_window, rand_window_batch

class DatasetGenerator:

    def __init__(self):

        # 1. Get positions directory
        print('Parsing:', config.fine_tuning_evaluations_dir)