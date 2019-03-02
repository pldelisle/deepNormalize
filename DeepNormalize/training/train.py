import tensorflow as tf
import argparse
import os.path
import time
import json
import numpy as np
import matplotlib.pyplot as plt

from DeepNormalize.io.data_provider import DataProvider
from DeepNormalize.utils.utils import export_inputs


class Trainer(object):

    def __init__(self, network, args, config):
        self.network = network
        self.config = config
        self.args = args
        self.learning_rate_node = None

    def _get_optimizer(self, training_iterations, global_step):
        self.learning_rate_node = tf.train.exponential_decay(learning_rate=self.config.get("model")["learning_rate"],
                                                             global_step=global_step,
                                                             decay_steps=training_iterations,
                                                             decay_rate=self.config.get("model")["learning_rate_decay"])
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate_node)

        return optimizer

    def _get_generators(self):
        # Instantiate both training and validation data sets.
        dataset_train = DataProvider(self.args.data_dir, "train", self.config.get("inputs"))
        dataset_validation = DataProvider(self.args.data_dir, "validation", self.config.get("inputs"))

        # Define training and validation datasets with the same structure.
        training_dataset = dataset_train.input()
        validation_dataset = dataset_validation.input()

        # A feedable iterator is defined by a handle placeholder and its structure. We
        # could use the `output_types` and `output_shapes` properties of either
        # `training_dataset` or `validation_dataset` here, because they have
        # identical structure.
        handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(
            handle, training_dataset.output_types, training_dataset.output_shapes)
        next_element = iterator.get_next()

        # You can use feedable iterators with a variety of different kinds of iterator
        # (such as one-shot and initializable iterators).
        training_iterator = training_dataset.make_one_shot_iterator()
        validation_iterator = validation_dataset.make_one_shot_iterator()

        return training_iterator, validation_iterator, next_element, handle

    def train(self):
        training_iterator, validation_iterator, next_element, handle = self._get_generators()

        # Session configuration.
        sess_config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=self.args.log_device_placement,
            intra_op_parallelism_threads=self.args.num_intra_threads,
            inter_op_parallelism_threads=self.args.num_inter_threads,
            gpu_options=tf.GPUOptions(force_gpu_compatible=True))

        sess = tf.Session(config=sess_config)

        # The `Iterator.string_handle()` method returns a tensor that can be evaluated
        # and used to feed the `handle` placeholder.
        training_handle = sess.run(training_iterator.string_handle())
        validation_handle = sess.run(validation_iterator.string_handle())

        # Loop forever, alternating between training and validation.
        while True:
            # Run 200 steps using the training dataset. Note that the training dataset is
            # infinite, and we resume from where we left off in the previous `while` loop
            # iteration.
            for _ in range(200):
                start_time = time.time()
                data = sess.run(next_element, feed_dict={handle: training_handle})
                if self.config.get("export_inputs"):
                    export_inputs(data)
                end_time = time.time()
                print("Training time " + str(end_time - start_time))
                start_time = time.time()
                validation_data = sess.run(next_element, feed_dict={handle: validation_handle})
                end_time = time.time()
                print("Validation time " + str(end_time - start_time))

            # Run one pass over the validation dataset.
            sess.run(validation_iterator.initializer)
            for _ in range(50):
                sess.run(next_element, feed_dict={handle: validation_handle})
