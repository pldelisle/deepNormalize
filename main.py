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
import json

from DeepNormalize.training.train_eager import Trainer
from DeepNormalize.model.deepNormalize import DeepNormalize

tf.enable_eager_execution()
tf.enable_v2_behavior()

print("TensorFlow version: {}".format(tf.__version__))
print("Keras version: {}".format(tf.keras.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))


def main(args, config):
    model = DeepNormalize(config=config, n_gpus=args.n_gpus)
    # model.compile(optimizer=tf.train.AdamOptimizer(0.001),
    #               loss="mse",
    #               metrics=["accuracy"])
    # try:
    #     model = tf.keras.utils.multi_gpu_model(model, gpus=args.n_gpus, cpu_merge=False)
    #     print("Training using multiple GPUs.")
    # except:
    #     print("Training using single GPU or CPU.")

    trainer = Trainer(model, args=args, config=config)

    trainer.run_eager_training()

    print("Debug")


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
        '--restore-dir',
        type=str,
        required=True,
        help='The directory where the model will be restored if exists.')
    parser.add_argument(
        '--variable-strategy',
        choices=['CPU', 'GPU'],
        type=str,
        default='CPU',
        help='Where to locate variable operations')
    parser.add_argument(
        '--n-gpus',
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

    if args.n_gpus <= 1:
        # Set visible device to desired GPU ID.
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # Parse the JSON configuration file.
    with open("config.json", "r") as json_file:
        config = json.loads(json_file.read())

    # Make some pre-testing on configuration.
    if args.n_gpus > 0:
        assert tf.test.is_gpu_available(), "Requested GPUs but none found."
    if args.n_gpus < 0:
        raise ValueError(
            'Invalid GPU count: \"--num-gpus\" must be 0 or a positive integer.')
    if args.n_gpus == 0 and args.variable_strategy == 'GPU':
        raise ValueError('num-gpus=0, CPU must be used as parameter server. Set'
                         '--variable-strategy=CPU.')
    if args.n_gpus != 0 and config.get("inputs")["subject_batch_size"] % args.n_gpus != 0:
        raise ValueError('--train-batch-size must be multiple of --n-gpus.')
    if args.n_gpus != 0 and config.get("inputs")["eval_batch_size"] % args.n_gpus != 0:
        raise ValueError('--eval-batch-size must be multiple of --n-gpus.')

    # Launch training.
    main(args=args, config=config)
