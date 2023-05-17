import tensorflow as tf


class HydraMLM(tf.keras.Model):

    # Metrics Functions
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        reduction=tf.keras.losses.Reduction.NONE
    )
    loss_tracker = tf.keras.metrics.Mean(name="loss")
    accuracy_tracker = tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")

    def train_step(self, inputs):
        features, labels, sample_weight, board = inputs

        with tf.GradientTape() as tape:
            predictions = self([board, features], training=True)
            loss = self.loss_fn(labels, predictions, sample_weight=sample_weight)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Compute our own metrics
        self.loss_tracker.update_state(loss, sample_weight=sample_weight)
        self.accuracy_tracker.update_state(labels, predictions, sample_weight=sample_weight)

        # Return a dict mapping metric names to current value
        return {"loss": self.loss_tracker.result(), "accuracy": self.accuracy_tracker.result()}

    def test_step(self, inputs):
        features, labels, sample_weight, board = inputs

        # Compute predictions
        predictions = self([board, features], training=False)

        # Compute the loss and update tracker
        loss = self.loss_fn(labels, predictions, sample_weight=sample_weight)
        self.loss_tracker.update_state(loss, sample_weight=sample_weight)

        # Update accuracy tracker
        self.accuracy_tracker.update_state(labels, predictions, sample_weight=sample_weight)

        # Return a dict mapping metric names to current value
        return {"loss": self.loss_tracker.result(), "accuracy": self.accuracy_tracker.result()}

    @property
    def metrics(self):
        return [self.loss_tracker, self.accuracy_tracker]


