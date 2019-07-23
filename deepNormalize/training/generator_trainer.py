#  -*- coding: utf-8 -*-
#  Copyright 2019 Pierre-Luc Delisle. All Rights Reserved.
#  #
#  Licensed under the MIT License;
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#      https://opensource.org/licenses/MIT
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ==============================================================================

import torch
import numpy as np

from samitorch.inputs.batch import Batch
from samitorch.logger.plots import ImagesPlot
from deepNormalize.logger.image_slicer import AdaptedImageSlicer
from deepNormalize.inputs.images import SliceType
from deepNormalize.training.base_model_trainer import DeepNormalizeModelTrainer

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")


class GeneratorTrainer(DeepNormalizeModelTrainer):

    def __init__(self, config, callbacks, discriminator_trainer, class_name):
        super(GeneratorTrainer, self).__init__(config, callbacks, class_name)
        self._discriminator_trainer = discriminator_trainer
        self._generated_images_plot = ImagesPlot(self._visdom, "Adapted Images")
        self._slicer = AdaptedImageSlicer()

    def train_batch(self, batch: Batch, detach=False):
        # Generate normalized data.
        generated_batch = self.predict(batch, detach=detach)

        # Loss measures generator's ability to fool the discriminator.
        loss_D_G_X_as_X = self.evaluate_discriminator_error_on_normalized_data(generated_batch)

        with amp.scale_loss(loss_D_G_X_as_X, self._config.optimizer) as scaled_loss:
            scaled_loss.backward(retain_graph=detach)

        self.step()

        return loss_D_G_X_as_X, generated_batch

    def evaluate_discriminator_error_on_normalized_data(self, generated_batch, detach=False):
        pred_D_G_X = self._discriminator_trainer.predict(generated_batch, detach=detach)

        y = torch.Tensor().new_tensor(
            data=np.zeros(shape=(generated_batch.x.size(0), )),
            dtype=torch.int8,
            device=self._config.running_config.device)
        pred_D_G_X.dataset_id = y

        loss_D_G_X_as_X = self._discriminator_trainer.evaluate_loss(
            torch.nn.functional.log_softmax(pred_D_G_X.x, dim=1),
            pred_D_G_X.dataset_id.long())

        pred_D_G_X.to_device('cpu')

        return -loss_D_G_X_as_X

    def train_generator_with_segmentation(self, batch, loss_S_G_X):
        generated_batch = self.predict(batch)
        loss_D_G_X_as_X = self.evaluate_discriminator_error_on_normalized_data(generated_batch)

        custom_loss = self._config.variables.alpha_ * loss_S_G_X.item() + self._config.variables.lambda_ * loss_D_G_X_as_X

        with amp.scale_loss(custom_loss, self._config.optimizer) as scaled_loss:
            scaled_loss.backward()

        self.step()

        return custom_loss, loss_D_G_X_as_X

    def update_image_plot(self, image):
        image = torch.nn.functional.interpolate(image, scale_factor=5, mode="trilinear", align_corners=True)
        self._generated_images_plot.update(self._slicer.get_slice(SliceType.AXIAL, image))

    def at_epoch_begin(self, epoch_num: torch.Tensor):
        super(GeneratorTrainer, self).at_epoch_begin()
        self.update_learning_rate_plot(epoch_num,
                                       torch.Tensor().new([self._config.optimizer.param_groups[0]['lr']]).cpu())
