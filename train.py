from hydra.HydraMLM import HydraMLM
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



def train():

    # --> Load Datasets
    training_dataset = tf.data.Dataset.load(os.path.join(config.datasets_dir, config.train_dataset))
    validation_dataset = tf.data.Dataset.load(os.path.join(config.datasets_dir, config.val_dataset))

    # --> Create Model
    model = build_model()

    # --> Train Model
    model_file = os.path.join(config.datasets_dir, config.model_name)
    checkpoint = ModelCheckpoint(model_file, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    plot_checkpoint = PlotCallback("hydra-mlm")
    history = model.fit(training_dataset, epochs=config.epochs, validation_data=validation_dataset, callbacks=[checkpoint, plot_checkpoint])
    print(history)
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
    output = hydra(board_inputs, move_inputs)

    # --> Hydra Model
    model = HydraMLM([board_inputs, move_inputs], output, name="hydra_mlm")

    # --> Compile Model
    optimizer = keras.optimizers.Adam()
    model.compile(optimizer=optimizer, jit_compile=False)

    # --> Save Model Details
    model.summary()
    model_img_file = os.path.join(config.models_dir, config.model_name + '.png')
    plot_model(model, to_file=model_img_file, show_shapes=True, show_layer_names=True, expand_nested=True)
    return model
















if __name__ == '__main__':
    train()



