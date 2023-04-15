import tensorflow as tf
import keras
from keras import layers
from keras.layers import TextVectorization
import pandas as pd
import numpy as np
import os
import pickle
import re
from keras.utils import plot_model
from hydra import config
from oldchess.Hydra import Hydra
from oldchess.HydraMLM import HydraMLM
from keras.callbacks import ModelCheckpoint


class MoveVectorizeLayer:


    def __init__(self):
        self.batch_size = config.batch_size
        self.max_len = config.seq_length


        self.root_dir = os.path.dirname(os.path.abspath(__file__))
        self.games_dir = os.path.join(self.root_dir, '../games')
        self.tokens_dir = os.path.join(self.root_dir, '../tokens')
        self.positions_dir = os.path.join(self.root_dir, '../positions')
        self.datasets_dir = os.path.join(self.root_dir, '../datasets')
        self.models_dir = os.path.join(self.root_dir, '../models')

        # --> 1. Get Vocabulary
        vocab_file = os.path.join(self.tokens_dir, 'tokens_1946.pkl')
        with open(vocab_file, 'rb') as f:
            self.vocab = list(pickle.load(f))
            self.vocab.sort()
        self.vocab_size = len(self.vocab)

        # --> 2. Get Positions Data
        # self.human_positions_file = os.path.join(self.positions_dir, 'human-training-positions-627.pkl')
        self.human_positions_file = os.path.join(self.positions_dir, 'human-training-positions-743847.pkl')
        self.human_positions = []
        with open(self.human_positions_file, 'rb') as f:
            self.human_positions = pickle.load(f)
        self.all_data = pd.DataFrame(self.human_positions)
        self.all_data = self.all_data[self.all_data['moves'] != '']
        self.all_data.reset_index(drop=True, inplace=True)
        print(self.all_data.head())
        print(self.all_data.shape)

        # --> 3. Split Data
        split_index = int(len(self.all_data) * 0.8)
        self.train_df = self.all_data.iloc[:split_index]
        self.test_df = self.all_data.iloc[split_index:]

        # --> 4. Create Vectorize Layer
        self.vectorize_layer = self.get_vectorize_layer(
            self.all_data.moves.values.tolist(),
            self.vocab_size,
            self.max_len,
            special_tokens=["[mask]"],
        )
        self.mask_token_id = self.vectorize_layer(["[mask]"]).numpy()[0][0]

        # --> 5. Prepare MLM Dataset
        self.all_moves = self.encode(self.all_data.moves.values.tolist())
        self.all_boards = self.all_data.board.values.tolist()
        split_idx = int(len(self.all_moves) * 0.8)
        self.train_moves = self.all_moves[:split_idx]
        self.train_boards = self.all_boards[:split_idx]
        self.validation_moves = self.all_moves[split_idx:]
        self.validation_boards = self.all_boards[split_idx:]


        train_x, train_y, train_sample_weights = self.get_masked_input_and_labels(self.train_moves)
        self.mlm_ds_train = tf.data.Dataset.from_tensor_slices(
            (train_x, train_y, train_sample_weights, self.train_boards)
        )
        self.mlm_ds_train = self.mlm_ds_train.shuffle(1000).batch(self.batch_size)

        validation_x, validation_y, validation_sample_weights = self.get_masked_input_and_labels(self.validation_moves)
        self.mlm_ds_validation = tf.data.Dataset.from_tensor_slices(
            (validation_x, validation_y, validation_sample_weights, self.validation_boards)
        )
        self.mlm_ds_validation = self.mlm_ds_validation.shuffle(1000).batch(self.batch_size)
        self.mlm_ds_validation.save("")
        # validation_data = tf.data.Dataset.from_tensor_slices(
        #     ((self.validation_boards, validation_x), validation_y, validation_sample_weights)
        # )
        # self.mlm_ds_validation = validation_data.shuffle(1000).batch(self.batch_size)

        x_masked_train, y_masked_labels, sample_weights = self.get_masked_input_and_labels(self.all_moves)
        self.mlm_ds = tf.data.Dataset.from_tensor_slices(
            (x_masked_train, y_masked_labels, sample_weights, self.all_boards)
        )
        self.mlm_ds = self.mlm_ds.shuffle(1000).batch(self.batch_size)

        # Shuffle the dataset

        # Calculate the number of samples for training and validation
        # train_size = int(0.8 * len(x_masked_train))  # 80% for training
        # validation_size = len(x_masked_train) - train_size  # 20% for validation

        # Split the dataset into training and validation datasets
        # self.train_mlm_ds = self.mlm_ds.take(train_size)
        # self.validation_mlm_ds = self.mlm_ds.skip(train_size)

        # Apply any additional transformations, such as batching
        # self.train_mlm_ds = self.train_mlm_ds.batch(self.batch_size)
        # self.validation_mlm_ds = self.validation_mlm_ds.batch(self.batch_size)

        # Save the datasets
        # datasets_dir = self.datasets_dir
        # train_mlm_ds_file = os.path.join(datasets_dir, 'train_mlm_dataset')
        # validation_mlm_ds_file = os.path.join(datasets_dir, 'validation_mlm_dataset')
        #
        # self.train_mlm_ds.save(train_mlm_ds_file)
        # self.validation_mlm_ds.save(validation_mlm_ds_file)



        # self.mlm_ds = self.mlm_ds.shuffle(1000).batch(self.batch_size)
        # self.mlm_ds_file = os.path.join(self.datasets_dir, 'mlm-dataset-72753')
        # self.mlm_ds.save(self.mlm_ds_file)

        # --> Encoding / Decoding Tokens
        self.id2token = dict(enumerate(self.vectorize_layer.get_vocabulary()))
        self.token2id = {y: x for x, y in self.id2token.items()}


        # --> Build Model
        self.hydra = Hydra(self.vocab_size)
        self.mlm_output = layers.Dense(self.vocab_size, name="move_prediction_head", activation="softmax")
        self.model = self.build_model()
        self.model.summary()
        model_img_file = os.path.join(self.models_dir, 'hydrachess.png')
        plot_model(self.model, to_file=model_img_file, show_shapes=True, show_layer_names=True, expand_nested=True)

        # --> Fit Model
        # self.model.fit(self.mlm_ds, epochs=5)
        model_file = os.path.join(self.models_dir, 'hydrachess')
        checkpoint = ModelCheckpoint(model_file, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
        self.model.fit(self.mlm_ds_train, epochs=5, validation_data=self.mlm_ds_validation, callbacks=[checkpoint])
        # self.model.save(model_file)


    def build_model(self):

        board_inputs = layers.Input(shape=(8, 8, 12,), name="board")
        move_inputs = layers.Input(shape=(128,), name="moves")
        # mlm_model = HydraModel([board_inputs, move_inputs], name="hydra_mlm")
        # # output = mlm_model(board_inputs, move_inputs)
        # # final_model = tf.keras.Model(inputs=[board_inputs, move_inputs], outputs=output)
        # mlm_model.compile(optimizer=keras.optimizers.Adam(), jit_compile=False)
        # return mlm_model


        # --> Hydra Encoder
        encoder_outputs = self.hydra(board_inputs, move_inputs)

        # --> Final Dense Prediction Head
        encoded_move_input = encoder_outputs[:, 1:, :]
        mlm_output = self.mlm_output(encoded_move_input)

        # --> Define Model
        mlm_model = HydraMLM([board_inputs, move_inputs], mlm_output, name="hydra_mlm")


        # --> Move Embeddings
        optimizer = keras.optimizers.Adam()
        mlm_model.compile(optimizer=optimizer, jit_compile=False)
        return mlm_model









    def custom_standardization(self, input_data):
        lowercase = tf.strings.lower(input_data)
        stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")
        return tf.strings.regex_replace(
            stripped_html, "[%s]" % re.escape("!#$%&'()*+,-./:;<=>?@\^_`{|}~"), ""
        )

    def get_vectorize_layer(self, texts, vocab_size, max_seq, special_tokens=["[MASK]"]):
        """Build Text vectorization layer

        Args:
          texts (list): List of string i.e. input texts
          vocab_size (int): vocab size
          max_seq (int): Maximum sequence lenght.
          special_tokens (list, optional): List of special tokens. Defaults to ['[MASK]'].

        Returns:
            layers.Layer: Return TextVectorization Keras Layer
        """
        vectorize_layer = TextVectorization(
            max_tokens=vocab_size,
            output_mode="int",
            standardize=self.custom_standardization,
            output_sequence_length=max_seq,
        )
        vectorize_layer.adapt(texts)

        # Insert mask token in vocabulary
        vocab = vectorize_layer.get_vocabulary()
        vocab = vocab[2: vocab_size - len(special_tokens)] + ["[mask]"]
        vectorize_layer.set_vocabulary(vocab)
        return vectorize_layer




    def encode(self, texts):
        encoded_texts = self.vectorize_layer(texts)
        return encoded_texts.numpy()





    def get_masked_input_and_labels(self, inputs):

        # encoded_texts: shape(N, 128) where 128 is the max sequence length
        # - filled with tokenized values
        encoded_texts = inputs


        # inp_mask: shape(N, 128)
        # - filled with True or False denoting whether to mask that token
        # - do not mask special tokens
        inp_mask = np.random.rand(*encoded_texts.shape) < 0.15
        inp_mask[encoded_texts <= 2] = False


        # labels: shape(N, 128)
        # - defines the labels for the masked tokens
        # - all unmasked token positions are set to -1 (ignore)
        labels = -1 * np.ones(encoded_texts.shape, dtype=int)
        labels[inp_mask] = encoded_texts[inp_mask]


        # encoded_texts_masked: shape(N, 128)
        # - modified copy of encoded_texts
        # - masked positions are replaced with mask token id
        encoded_texts_masked = np.copy(encoded_texts)

        # - 90% of masked tokens are predicted
        predict_prob = 1.0
        inp_mask_2mask = inp_mask & (np.random.rand(*encoded_texts.shape) < predict_prob)
        encoded_texts_masked[inp_mask_2mask] = self.mask_token_id  # mask token is the last in the dict

        # - 10% of masked tokens are replaced with random token
        random_prob = 0.0
        inp_mask_2random = inp_mask_2mask & (np.random.rand(*encoded_texts.shape) < random_prob)
        encoded_texts_masked[inp_mask_2random] = np.random.randint(3, self.mask_token_id, inp_mask_2random.sum())

        # sample_weights: shape(N, 128)
        # - defines the weights for the loss function
        # - weights are 0 for all unmasked token positions and 1 for masked token positions
        sample_weights = np.ones(labels.shape)
        sample_weights[labels == -1] = 0

        # y_labels: shape(N, 128)
        # - defines the labels for the masked tokens
        # - is essentially the input to this function
        y_labels = np.copy(encoded_texts)

        return encoded_texts_masked, y_labels, sample_weights


if __name__ == '__main__':
    move_vectorize_layer = MoveVectorizeLayer()



