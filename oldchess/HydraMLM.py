
from keras import Model
from keras import layers
import keras
import pickle
import tensorflow as tf
import os
from keras_nlp.layers import TransformerEncoder, TokenAndPositionEmbedding, MaskedLMHead
from TokenizerManager import TokenizerManager

# --> MLM Loss
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
    reduction=tf.keras.losses.Reduction.NONE
)
loss_tracker = tf.keras.metrics.Mean(name="loss")

accuracy_tracker = tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")

class HydraMLM(tf.keras.Model):

    def train_step(self, inputs):
        features, labels, sample_weight, board = inputs

        with tf.GradientTape() as tape:
            predictions = self([board, features], training=True)
            loss = loss_fn(labels, predictions, sample_weight=sample_weight)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Compute our own metrics
        loss_tracker.update_state(loss, sample_weight=sample_weight)
        accuracy_tracker.update_state(labels, predictions, sample_weight=sample_weight)

        # Return a dict mapping metric names to current value
        return {"loss": loss_tracker.result(), "accuracy": accuracy_tracker.result()}

    def test_step(self, inputs):
        features, labels, sample_weight, board = inputs

        # Compute predictions
        predictions = self([board, features], training=False)

        # Compute the loss
        loss = loss_fn(labels, predictions, sample_weight=sample_weight)

        # Update the loss tracker and accuracy tracker
        loss_tracker.update_state(loss, sample_weight=sample_weight)
        accuracy_tracker.update_state(labels, predictions, sample_weight=sample_weight)

        # Return a dict mapping metric names to current value
        return {"loss": loss_tracker.result(), "accuracy": accuracy_tracker.result()}

    @property
    def metrics(self):
        return [loss_tracker, accuracy_tracker]

    # def test_step(self, inputs):
    #     features, labels, sample_weight, board = inputs
    #
    #     # Compute predictions
    #     predictions = self([board, features], training=False)
    #
    #     # Compute the loss
    #     loss = loss_fn(labels, predictions, sample_weight=sample_weight)
    #
    #     # Update the loss tracker
    #     loss_tracker.update_state(loss, sample_weight=sample_weight)
    #
    #     # Return a dict mapping metric names to current value
    #     return {"loss": loss_tracker.result()}
    #
    # def train_step(self, inputs):
    #     features, labels, sample_weight, board = inputs
    #
    #     with tf.GradientTape() as tape:
    #         predictions = self([board, features], training=True)
    #         loss = loss_fn(labels, predictions, sample_weight=sample_weight)
    #
    #     # Compute gradients
    #     trainable_vars = self.trainable_variables
    #     gradients = tape.gradient(loss, trainable_vars)
    #
    #     # Update weights
    #     self.optimizer.apply_gradients(zip(gradients, trainable_vars))
    #
    #     # Compute our own metrics
    #     loss_tracker.update_state(loss, sample_weight=sample_weight)
    #
    #     # Return a dict mapping metric names to current value
    #     return {"loss": loss_tracker.result()}
    #
    # @property
    # def metrics(self):
    #     return [loss_tracker]