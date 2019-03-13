# -*- coding: utf-8 -*-
# Copyright 2019 Pierre-Luc Delisle. All Rights Reserved.
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
A class for providing data to a PyTorch neural network.

"""

import torch.utils.data
import torchvision

from DeepNormalize.io.dataset import MultimodalNiftiDataset
from DeepNormalize.io.transforms import *


class DataProvider(object):

    def __init__(self, config):
        """
        Data provider for multimodal MRI.
        :param config: A JSON configuration file for input parameters.
        """

        self._MRBrainS = config.get("dataset").get("MRBrainS")
        self._iSEG = config.get("dataset").get("iSEG")

        self._transform = torchvision.transforms.Compose([RandomCrop3D(config.get("patch_size")), ToTensor()])

        self._MRBrainS_dataset = MultimodalNiftiDataset([self._MRBrainS.get("image_t1_path"),
                                                         self._MRBrainS.get("image_t2_path")],
                                                        [self._MRBrainS.get("label_path")],
                                                        self._transform)

        self._iSEG_dataset = MultimodalNiftiDataset([self._iSEG.get("image_t1_path"),
                                                     self._iSEG.get("image_t2_path")],
                                                    [self._iSEG.get("label_path")],
                                                    self._transform)

        self._training_batch_size = config.get("training_batch_size")
        self._valid_batch_size = config.get("valid_batch_size")

        # Declare a Sampler object here if required.

    def get_data_loader(self):
        MRBrainS_training, MRBRainS_valid = torch.utils.data.random_split(self._MRBrainS_dataset, [4, 1])
        iSEG_training, iSEG_valid = torch.utils.data.random_split(self._iSEG_dataset, [8, 2])

        training_dataset = torch.utils.data.ConcatDataset((MRBrainS_training, iSEG_training))
        validation_dataset = torch.utils.data.ConcatDataset((MRBRainS_valid, iSEG_valid))

        training_loader = torch.utils.data.DataLoader(training_dataset,
                                                      batch_size=self._training_batch_size,
                                                      shuffle=True,
                                                      num_workers=4,
                                                      pin_memory=torch.cuda.is_available())
        validation_loader = torch.utils.data.DataLoader(validation_dataset,
                                                        batch_size=self._valid_batch_size,
                                                        shuffle=True,
                                                        num_workers=4,
                                                        pin_memory=torch.cuda.is_available())

        return training_loader, validation_loader
