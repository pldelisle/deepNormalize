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

import argparse
import os.path
import json
import sys
import numpy as np
import torch
import torchsummary

from matplotlib import __version__ as mplver

from DeepNormalize.io.data_provider import DataProvider

from DeepNormalize.training.train import Trainer

from DeepNormalize.model.DeepNormalize.deepNormalize import DeepNormalize
from DeepNormalize.model.GAN.gan import Discriminator

pv = sys.version_info
print('Python version: {}.{}.{}'.format(pv.major, pv.minor, pv.micro))
print('Numpy version: {}'.format(np.__version__))
print("PyTorch version: {}".format(torch.__version__))
print('Matplotlib version: {}'.format(mplver))


def main(args, config):
    # print(Cuda())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator = DeepNormalize(config=config.get("model").get("deep_normalize"))
    torchsummary.summary(generator, input_size=(2, 64, 64, 64), device="cpu")

    discriminator = Discriminator()
    torchsummary.summary(discriminator, input_size=(2, 64, 64, 64), device="cpu")

    if torch.cuda.device_count() > 1:
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        print("Let's use", torch.cuda.device_count(), "GPUs.")
        generator = torch.nn.DataParallel(generator)
        discriminator = torch.nn.DataParallel(discriminator)
    else:
        generator = generator.to(device).float()
        discriminator = discriminator.to(device).float()
        print("Training on single GPU or CPU.")

    data_provider = DataProvider(config=config.get("inputs"))
    trainer = Trainer(generator,
                      discriminator,
                      data_provider,
                      config_generator=config.get("model").get("deep_normalize").get("training"),
                      config_discriminator=config.get("model").get("resnet").get("training"))

    trainer.train()

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
        required=False,
        help='The directory where the model will be restored if exists.')
    parser.add_argument(
        '--n-gpus',
        type=int,
        default=1,
        help='The number of gpus used. Uses only CPU if set to 0.')
    parser.add_argument(
        '--gpu-id',
        type=str,
        default="0",
        help='The GPU ID on which to run the training.')
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

    args = parser.parse_args()

    if args.n_gpus <= 1:
        # Set visible device to desired GPU ID.
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # Parse the JSON configuration file.
    with open("config.json", "r") as json_file:
        config = json.loads(json_file.read())

    # Make some pre-testing on configuration.
    if args.n_gpus > 0:
        assert torch.cuda.is_available(), "Requested GPUs but none found."

    # Launch training.
    main(args=args, config=config)
