# Copyright 2018 Pierre-Luc Delisle. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
A binary to train using multiple GPUs with synchronous updates.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import argparse
import os

from datetime import datetime
import os.path
import re
import time
import json

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

from DeepNormalize.io.data_provider import DataProvider
from DeepNormalize.utils import preprocessing
from DeepNormalize.utils.sampler import Sampler

# def tower_loss(scope, images, labels):
#     """Calculate the total loss on a single tower running the DeepNormalize model.
#
#     Args:
#       scope: unique prefix string identifying the DeepNormalize tower, e.g. 'tower_0'
#       images: Images. 5D tensor of shape [batch_size, height, width, depth, 1].
#       labels: Labels. 5D tensor of shape [batch_size, height, width, depth, 1].
#
#     Returns:
#        Tensor of shape [] containing the total loss for a batch of data
#     """
#
#     # Build inference Graph.
#     logits = cifar10.inference(images)
#
#     # Build the portion of the Graph calculating the losses. Note that we will
#     # assemble the total_loss using a custom function below.
#     _ = cifar10.loss(logits, labels)
#
#     # Assemble all of the losses for the current tower only.
#     losses = tf.get_collection('losses', scope)
#
#     # Calculate the total loss for the current tower.
#     total_loss = tf.add_n(losses, name='total_loss')
#
#     # Attach a scalar summary to all individual losses and the total loss; do the
#     # same for the averaged version of the losses.
#     for l in losses + [total_loss]:
#         # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
#         # session. This helps the clarity of presentation on tensorboard.
#         loss_name = re.sub('%s_[0-9]*/' % cifar10.TOWER_NAME, '', l.op.name)
#         tf.summary.scalar(loss_name, l)
#
#     return total_loss
#
#
# def average_gradients(tower_grads):
#     """Calculate the average gradient for each shared variable across all towers.
#
#     Note that this function provides a synchronization point across all towers.
#
#     Args:
#       tower_grads: List of lists of (gradient, variable) tuples. The outer list
#         is over individual gradients. The inner list is over the gradient
#         calculation for each tower.
#     Returns:
#        List of pairs of (gradient, variable) where the gradient has been averaged
#        across all towers.
#     """
#     average_grads = []
#     for grad_and_vars in zip(*tower_grads):
#         # Note that each grad_and_vars looks like the following:
#         #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
#         grads = []
#         for g, _ in grad_and_vars:
#             # Add 0 dimension to the gradients to represent the tower.
#             expanded_g = tf.expand_dims(g, 0)
#
#             # Append on a 'tower' dimension which we will average over below.
#             grads.append(expanded_g)
#
#         # Average over the 'tower' dimension.
#         grad = tf.concat(axis=0, values=grads)
#         grad = tf.reduce_mean(grad, 0)
#
#         # Keep in mind that the Variables are redundant because they are shared
#         # across towers. So .. we will just return the first tower's pointer to
#         # the Variable.
#         v = grad_and_vars[0][1]
#         grad_and_var = (grad, v)
#         average_grads.append(grad_and_var)
#     return average_grads
#
#
# def train(argv):
#     """Train CIFAR-10 for a number of steps."""
#     with tf.Graph().as_default(), tf.device('/cpu:0'):
#         # Create a variable to count the number of train() calls. This equals the
#         # number of batches processed * FLAGS.num_gpus.
#         global_step = tf.get_variable(
#             'global_step', [],
#             initializer=tf.constant_initializer(0), trainable=False)
#
#         # Calculate the learning rate schedule.
#         num_batches_per_epoch = (cifar10.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN /
#                                  FLAGS.batch_size / FLAGS.num_gpus)
#         decay_steps = int(num_batches_per_epoch * cifar10.NUM_EPOCHS_PER_DECAY)
#
#         # Decay the learning rate exponentially based on the number of steps.
#         lr = tf.train.exponential_decay(cifar10.INITIAL_LEARNING_RATE,
#                                         global_step,
#                                         decay_steps,
#                                         cifar10.LEARNING_RATE_DECAY_FACTOR,
#                                         staircase=True)
#
#         # Create an optimizer that performs gradient descent.
#         opt = tf.train.GradientDescentOptimizer(lr)
#
#         # Get images and labels for CIFAR-10.
#         images, labels = cifar10.distorted_inputs()
#         batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue(
#             [images, labels], capacity=2 * FLAGS.num_gpus)
#         # Calculate the gradients for each model tower.
#         tower_grads = []
#         with tf.variable_scope(tf.get_variable_scope()):
#             for i in xrange(FLAGS.num_gpus):
#                 with tf.device('/gpu:%d' % i):
#                     with tf.name_scope('%s_%d' % (cifar10.TOWER_NAME, i)) as scope:
#                         # Dequeues one batch for the GPU
#                         image_batch, label_batch = batch_queue.dequeue()
#                         # Calculate the loss for one tower of the CIFAR model. This function
#                         # constructs the entire CIFAR model but shares the variables across
#                         # all towers.
#                         loss = tower_loss(scope, image_batch, label_batch)
#
#                         # Reuse variables for the next tower.
#                         tf.get_variable_scope().reuse_variables()
#
#                         # Retain the summaries from the final tower.
#                         summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
#
#                         # Calculate the gradients for the batch of data on this CIFAR tower.
#                         grads = opt.compute_gradients(loss)
#
#                         # Keep track of the gradients across all towers.
#                         tower_grads.append(grads)
#
#         # We must calculate the mean of each gradient. Note that this is the
#         # synchronization point across all towers.
#         grads = average_gradients(tower_grads)
#
#         # Add a summary to track the learning rate.
#         summaries.append(tf.summary.scalar('learning_rate', lr))
#
#         # Add histograms for gradients.
#         for grad, var in grads:
#             if grad is not None:
#                 summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))
#
#         # Apply the gradients to adjust the shared variables.
#         apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
#
#         # Add histograms for trainable variables.
#         for var in tf.trainable_variables():
#             summaries.append(tf.summary.histogram(var.op.name, var))
#
#         # Track the moving averages of all trainable variables.
#         variable_averages = tf.train.ExponentialMovingAverage(
#             cifar10.MOVING_AVERAGE_DECAY, global_step)
#         variables_averages_op = variable_averages.apply(tf.trainable_variables())
#
#         # Group all updates to into a single train op.
#         train_op = tf.group(apply_gradient_op, variables_averages_op)
#
#         # Create a saver.
#         saver = tf.train.Saver(tf.global_variables())
#
#         # Build the summary operation from the last tower summaries.
#         summary_op = tf.summary.merge(summaries)
#
#         # Build an initialization operation to run below.
#         init = tf.global_variables_initializer()
#
#         # Start running operations on the Graph. allow_soft_placement must be set to
#         # True to build towers on GPU, as some of the ops do not have GPU
#         # implementations.
#         sess = tf.Session(config=tf.ConfigProto(
#             allow_soft_placement=True,
#             log_device_placement=FLAGS.log_device_placement))
#         sess.run(init)
#
#         # Start the queue runners.
#         tf.train.start_queue_runners(sess=sess)
#
#         summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)
#
#         for step in xrange(FLAGS.max_steps):
#             start_time = time.time()
#             _, loss_value = sess.run([train_op, loss])
#             duration = time.time() - start_time
#
#             assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
#
#             if step % 10 == 0:
#                 num_examples_per_step = FLAGS.batch_size * FLAGS.num_gpus
#                 examples_per_sec = num_examples_per_step / duration
#                 sec_per_batch = duration / FLAGS.num_gpus
#
#                 format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
#                               'sec/batch)')
#                 print(format_str % (datetime.now(), step, loss_value,
#                                     examples_per_sec, sec_per_batch))
#
#             if step % 100 == 0:
#                 summary_str = sess.run(summary_op)
#                 summary_writer.add_summary(summary_str, step)
#
#             # Save the model checkpoint periodically.
#             if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
#                 checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
#                 saver.save(sess, checkpoint_path, global_step=step)


def train(args, config):
	# Session configuration.
	sess_config = tf.ConfigProto(
		allow_soft_placement=True,
		log_device_placement=args.log_device_placement,
		intra_op_parallelism_threads=args.num_intra_threads,
		inter_op_parallelism_threads=args.num_inter_threads,
		gpu_options=tf.GPUOptions(force_gpu_compatible=True))

	dataset_train = DataProvider(args.data_dir, "train", config)
	dataset_validation = DataProvider(args.data_dir, "validation", config)

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
	validation_iterator = validation_dataset.make_initializable_iterator()

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
			end_time = time.time()
			print(end_time-start_time)
		# Run one pass over the validation dataset.
		sess.run(validation_iterator.initializer)
		for _ in range(50):
			sess.run(next_element, feed_dict={handle: validation_handle})


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--data-dir',
		type=str,
		required=True,
		help='The directory where the deepNormalize input data is stored.')
	parser.add_argument(
		'--job-dir',
		type=str,
		required=True,
		help='The directory where the model will be stored.')
	parser.add_argument(
		'--variable-strategy',
		choices=['CPU', 'GPU'],
		type=str,
		default='CPU',
		help='Where to locate variable operations')
	parser.add_argument(
		'--num-gpus',
		type=int,
		default=2,
		help='The number of gpus used. Uses only CPU if set to 0.')
	parser.add_argument(
		'--train-steps',
		type=int,
		default=1000000,
		help='The number of steps to use for training.')
	parser.add_argument(
		'--train-batch-size',
		type=int,
		default=32,
		help='Batch size for training.')
	parser.add_argument(
		'--eval-batch-size',
		type=int,
		default=32,
		help='Batch size for validation.')
	parser.add_argument(
		'--weight-decay',
		type=float,
		default=2e-4,
		help='Weight decay for convolutions.')
	parser.add_argument(
		'--learning-rate',
		type=float,
		default=0.0001,
		help="""\
      The inital learning rate value. The learning rate will decrease
      during training. For more details check the model_fn implementation in
      this file.\
      """)
	parser.add_argument(
		'--sync',
		action='store_true',
		default=False,
		help="""\
      If present when running in a distributed environment will run on sync mode.\
      """)
	parser.add_argument(
		'--num-intra-threads',
		type=int,
		default=4,
		help="""\
      Number of threads to use for intra-op parallelism. When training on CPU
      set to 0 to have the system pick the appropriate number or alternatively
      set it to the number of physical CPU cores.\
      """)
	parser.add_argument(
		'--num-inter-threads',
		type=int,
		default=4,
		help="""\
      Number of threads to use for inter-op parallelism. If set to 0, the
      system will pick an appropriate number.\
      """)
	parser.add_argument(
		'--data-format',
		type=str,
		default=None,
		help="""\
      If not set, the data format best for the training device is used. 
      Allowed values: channels_first (NCHW) channels_last (NHWC).\
      """)
	parser.add_argument(
		'--log-device-placement',
		action='store_true',
		default=False,
		help='Whether to log device placement.')

	args = parser.parse_args()

	if args.num_gpus > 0:
		assert tf.test.is_gpu_available(), "Requested GPUs but none found."
	if args.num_gpus < 0:
		raise ValueError(
			'Invalid GPU count: \"--num-gpus\" must be 0 or a positive integer.')
	if args.num_gpus == 0 and args.variable_strategy == 'GPU':
		raise ValueError('num-gpus=0, CPU must be used as parameter server. Set'
						 '--variable-strategy=CPU.')
	if args.num_gpus != 0 and args.train_batch_size % args.num_gpus != 0:
		raise ValueError('--train-batch-size must be multiple of --num-gpus.')
	if args.num_gpus != 0 and args.eval_batch_size % args.num_gpus != 0:
		raise ValueError('--eval-batch-size must be multiple of --num-gpus.')

	with open("config.json", "r") as json_file:
		config = json.loads(json_file.read())

	train(args=args, config=config)
