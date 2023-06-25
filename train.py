from hydra.HydraMLM import HydraMLM
from hydra.HydraPredict import HydraPredict
from hydra.Hydra import Hydra
from keras import layers
import keras
import os
from hydra import config
from keras.utils import plot_model
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import time


class PlotCallback(tf.keras.callbacks.Callback):
    def __init__(self, name, plot_dir='plots'):
        super(PlotCallback, self).__init__()
        self.plot_dir = plot_dir
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)
        self.train_accuracies = []
        self.val_accuracies = []
        self.plot_name = name
        self.time = time.time()
        self.num_positions = config.train_dataset.split('-')[-1]

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}

        train_acc = logs.get('accuracy')
        val_acc = logs.get('val_accuracy')

        if train_acc is not None:
            self.train_accuracies.append(train_acc)
        if val_acc is not None:
            self.val_accuracies.append(val_acc)

        plt.close()
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_accuracies, label='Training Accuracy')
        plt.plot(self.val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title(f'Model: {self.plot_name} - Dataset Positions {self.num_positions} - Epoch {epoch + 1}')
        plt.legend()
        plt.savefig(os.path.join(self.plot_dir, f'{self.plot_name}-{self.num_positions}-{self.time}.png'))
        plt.show()


from preprocess.DatasetGenerator import DatasetGenerator
from preprocess.FineTuningDatasetGenerator import FineTuningDatasetGenerator

def train():

    #####################
    ### Load Datasets ###
    #####################
    dataset_generator = DatasetGenerator()

    # --> Interleave Datasets
    # training_dataset, validation_dataset = dataset_generator.get_interleave_dataset()

    # --> Load Datasets
    training_dataset, validation_dataset = DatasetGenerator.load_datasets()
    print('Finished loading datasets...')

    # --> Create Model
    model = build_model()

    # --> Train Model
    model_file = os.path.join(config.datasets_dir, config.model_name)
    checkpoint = ModelCheckpoint(model_file, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    plot_checkpoint = PlotCallback("hydra-mlm")
    history = model.fit(training_dataset, epochs=config.epochs, validation_data=validation_dataset, callbacks=[checkpoint, plot_checkpoint])

    # --> Plot Training History
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()






def build_model():

    # --> Inputs
    board_inputs = layers.Input(shape=(8, 8, 12,), name="board")
    move_inputs = layers.Input(shape=(config.seq_length,), name="moves")

    # --> Hydra Encoder
    hydra = Hydra()
    # output = hydra.call_old(board_inputs, move_inputs)
    output = hydra(board_inputs, move_inputs)

    # --> Hydra Model
    model = HydraMLM([board_inputs, move_inputs], output, name="hydra_mlm")

    # --> Compile Model
    # optimizer = tf.keras.optimizers.Adam()
    optimizer = tf.keras.optimizers.legacy.Adam()
    model.compile(optimizer=optimizer, jit_compile=False)

    # --> Save Model Details
    model.summary(expand_nested=True)
    model_img_file = os.path.join(config.models_dir, config.model_name + '.png')
    plot_model(model, to_file=model_img_file, show_shapes=True, show_layer_names=True, expand_nested=False)
    return model






def fine_tune():
    #####################
    ### Load Datasets ###
    #####################
    # dataset_generator = FineTuningDatasetGenerator()
    # training_dataset, validation_dataset = dataset_generator.get_datasets()

    training_dataset, validation_dataset = FineTuningDatasetGenerator.load_datasets()
    print('Finished loading datasets...')

    model = build_fine_tuning()
    print('Finished building model...')

    # --> Train Model
    model_file = os.path.join(config.datasets_dir, config.model_name)
    checkpoint = ModelCheckpoint(model_file, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    plot_checkpoint = PlotCallback("hydra-ft")
    history = model.fit(training_dataset, epochs=config.epochs, validation_data=validation_dataset,
                        callbacks=[checkpoint, plot_checkpoint])

    # --> Plot Training History
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()




def build_fine_tuning():

    # --> Inputs
    board_inputs = layers.Input(shape=(8, 8, 12,), name="board")
    move_inputs = layers.Input(shape=(config.seq_length,), name="moves")

    # --> Hydra Encoder
    hydra = Hydra()
    # output = hydra.call_old(board_inputs, move_inputs)
    output = hydra(board_inputs, move_inputs)

    # --> Hydra Model
    model = HydraPredict([board_inputs, move_inputs], output, name="hydra_predict")

    # --> Compile Model
    optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer=optimizer, jit_compile=False)

    # --> Save Model Details
    model.summary(expand_nested=True)
    model_img_file = os.path.join(config.models_dir, config.model_name + '-ft.png')
    plot_model(model, to_file=model_img_file, show_shapes=True, show_layer_names=True, expand_nested=False)
    return model



if __name__ == '__main__':
    # train()
    fine_tune()



