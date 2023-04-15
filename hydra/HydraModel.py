import tensorflow as tf
from keras import layers
import keras


# --> Custom Layers
from hydra.layers.MoveEmbedding import MoveEmbedding
from hydra.layers.BoardEmbedding import BoardEmbedding
from hydra.layers.ModalityFusion import ModalityFusion
from hydra.layers.Encoder import Encoder

# --> Output Heads
from hydra.heads.MovePrediction import MovePrediction

from hydra.heads.MoveMaskPrediction import MoveMaskPrediction



# --> MLM Loss
loss_fn = keras.losses.SparseCategoricalCrossentropy(
    reduction=tf.keras.losses.Reduction.NONE
)
loss_tracker = keras.metrics.Mean(name="loss")





class HydraModel(tf.keras.Model):

    def __init__(self, *args, **kwargs):
        super(HydraModel, self).__init__(*args, **kwargs)
        self.mode = 'pretrain'

        # --> Move Embedding
        self.move_embedding = MoveEmbedding()

        # --> Board Embedding
        self.board_embedding = BoardEmbedding()

        # --> Modality Fusion
        self.modality_fusion = ModalityFusion()

        # --> Encoder
        self.encoder = Encoder()

        # --> Output Heads
        self.move_prediction_head = MovePrediction()
        self.move_mask_prediction_head = MoveMaskPrediction()

    def __call__(self, board_inputs, move_inputs, mask=None):

        # --> Board Embedding
        board_embedding = self.board_embedding(board_inputs)

        # --> Move Embedding
        move_embedding = self.move_embedding(move_inputs)

        # --> Combine Board and Move Embeddings
        encoder_inputs = self.modality_fusion(board_embedding, move_embedding)

        # --> Encoder Stack
        encoder_outputs = self.encoder(encoder_inputs)

        # --> Output Heads
        output = []
        if self.mode == 'pretrain':
            output = self.move_mask_prediction_head(encoder_outputs)
        elif self.mode == 'predict':
            output = self.move_prediction_head(encoder_outputs)

        return output


    def train_step(self, inputs):
        if self.mode == 'pretrain':
            return self.pretrain_step(inputs)


    def pretrain_step(self, inputs):
        features, labels, sample_weight, board = inputs

        with tf.GradientTape() as tape:
            predictions = self([board, features])
            loss = loss_fn(labels, predictions, sample_weight=sample_weight)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Compute our own metrics
        loss_tracker.update_state(loss, sample_weight=sample_weight)

        # Return a dict mapping metric names to current value
        return {"loss": loss_tracker.result()}


    @property
    def metrics(self):
        return [loss_tracker]



















