import tensorflow as tf
import time
import logging

from DeepNormalize.layers.loss import tversky, dice_coefficient
from DeepNormalize.layers.tversky_loss import TverskyLoss
from DeepNormalize.io.data_provider import DataProvider
from DeepNormalize.utils.utils import export_inputs


class Trainer(object):

    def __init__(self, model, args, config):
        self.model = model
        self.config = config
        self.args = args
        self.loss = TverskyLoss(n_classes=config.get("model")["n_classes"], alpha=0.3, beta=0.7)

    def _get_generators(self):
        # Instantiate both training and validation data sets.
        dataset_train = DataProvider(self.args.data_dir, "train", self.config.get("inputs"))
        dataset_validation = DataProvider(self.args.data_dir, "validation", self.config.get("inputs"))

        # Define training and validation datasets with the same structure.
        training_dataset = dataset_train.input()
        validation_dataset = dataset_validation.input()

        training_iterator = training_dataset.make_one_shot_iterator()
        validation_iterator = validation_dataset.make_one_shot_iterator()

        return training_iterator, validation_iterator

    @staticmethod
    def compute_accuracy(logits, labels):
        predictions = tf.argmax(logits, axis=1, output_type=tf.int32)
        labels = tf.squeeze(tf.cast(labels, tf.int32), axis=1)
        batch_size = int(logits.shape[0] * logits.shape[2] * logits.shape[3] * logits.shape[4])
        return tf.reduce_sum(
            tf.cast(tf.equal(predictions, labels), dtype=tf.float32)) / batch_size

    def train(self, model, optimizer, dataset, step_counter, log_interval):

        """Trains model on `dataset` using `optimizer`."""

        start = time.time()
        for (batch, (images, _, labels)) in enumerate(dataset):
            # with tf.contrib.summary.record_summaries_every_n_global_steps(
            #         10, global_step=step_counter):
            # Record the operations used to compute the loss given the input,
            # so that the gradient of the loss with respect to the variables
            # can be computed.
            with tf.GradientTape() as tape:
                logits = model(images, training=True)
                loss_value = self.loss.tversky(logits, labels)
                batch_accuracy = self.compute_accuracy(logits, labels).numpy()
            grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(
                zip(grads, model.trainable_variables), global_step=step_counter)
            if log_interval and batch % log_interval == 0:
                rate = log_interval / (time.time() - start)
                print('Step #%d\tLoss: %.6f (%d steps/sec)' % (batch, loss_value, rate))
                start = time.time()

    def run_eager_training(self):
        training_iterator, validation_iterator = self._get_generators()

        # Declare a global step variable.
        step_counter = tf.train.get_or_create_global_step()

        learning_rate_node = tf.train.exponential_decay(learning_rate=self.config.get("model")["learning_rate"],
                                                        global_step=step_counter,
                                                        decay_steps=10000,
                                                        decay_rate=self.config.get("model")["learning_rate_decay"],
                                                        staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_node)

        # Train and evaluate for a set number of epochs.

        for _ in range(self.config.get("model")["n_epochs"]):
            start = time.time()
            # with summary_writer.as_default():
            self.train(self.model, optimizer, training_iterator, step_counter, self.config.get("model")["log_interval"])
            end = time.time()
            # print('\nTrain time for epoch #%d (%d total steps): %f' %
            #       (checkpoint.save_counter.numpy() + 1,
            #        step_counter.numpy(),
            #        end - start))
            # with test_summary_writer.as_default():
            #     test(model, test_ds)
            # checkpoint.save(checkpoint_prefix)
