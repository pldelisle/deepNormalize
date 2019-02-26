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
import os.path
import time
import json
import numpy as np
import matplotlib.pyplot as plt

from DeepNormalize.io.data_provider import DataProvider
from DeepNormalize.utils.utils import print_inputs


def train(args, config):
    verbose = False
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
    validation_iterator = validation_dataset.make_one_shot_iterator()

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
            if verbose:
                print_inputs(data)
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
        default=1,
        help='The number of gpus used. Uses only CPU if set to 0.')
    parser.add_argument(
        '--gpu-id',
        type=str,
        default="1",
        help='The GPU ID on which to run the training.')
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

    # Set visible device to desired GPU ID.
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # Parse the JSON configuration file.
    with open("config.json", "r") as json_file:
        config = json.loads(json_file.read())

    # Make some pre-testing on configuration.
    if args.num_gpus > 0:
        assert tf.test.is_gpu_available(), "Requested GPUs but none found."
    if args.num_gpus < 0:
        raise ValueError(
            'Invalid GPU count: \"--num-gpus\" must be 0 or a positive integer.')
    if args.num_gpus == 0 and args.variable_strategy == 'GPU':
        raise ValueError('num-gpus=0, CPU must be used as parameter server. Set'
                         '--variable-strategy=CPU.')
    if args.num_gpus != 0 and config.get("subject_batch_size") % args.num_gpus != 0:
        raise ValueError('--train-batch-size must be multiple of --num-gpus.')
    if args.num_gpus != 0 and config.get("eval_batch_size") % args.num_gpus != 0:
        raise ValueError('--eval-batch-size must be multiple of --num-gpus.')

    # Launch training.
    train(args=args, config=config)
