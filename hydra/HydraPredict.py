import tensorflow as tf
import tensorflow_ranking as tfr


class HydraPredict(tf.keras.Model):

    loss_fn = tfr.keras.losses.ApproxNDCGLoss(name='loss')
    # loss_fn = tfr.losses.make_loss_fn(tfr.losses.RankingLossKey.APPROX_NDCG_LOSS, name='loss')

    precision_tracker = tfr.keras.metrics.PrecisionMetric(name="accuracy", topn=3)
    # precision_tracker = tfr.metrics.make_ranking_metric_fn(tfr.metrics.RankingMetricKey.PRECISION, name="accuracy")

    loss_tracker = tf.keras.metrics.Mean(name="loss")


    def train_step(self, inputs):
        previous_moves, relevancy_scores, board_tensor = inputs

        with tf.GradientTape() as tape:
            predictions = self([board_tensor, previous_moves], training=True)
            loss = self.loss_fn(relevancy_scores, predictions)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Compute our own metrics
        self.loss_tracker.update_state(loss)
        self.precision_tracker.update_state(relevancy_scores, predictions)

        # Return a dict mapping metric names to current value
        return {"loss": self.loss_tracker.result(), "accuracy": self.precision_tracker.result()}

    def test_step(self, inputs):
        previous_moves, relevancy_scores, board_tensor = inputs
        predictions = self([board_tensor, previous_moves], training=False)
        loss = self.loss_fn(relevancy_scores, predictions)
        self.loss_tracker.update_state(loss)
        self.precision_tracker.update_state(relevancy_scores, predictions)
        return {"loss": self.loss_tracker.result(), "accuracy": self.precision_tracker.result()}

    @property
    def metrics(self):
        return [self.loss_tracker, self.precision_tracker]



