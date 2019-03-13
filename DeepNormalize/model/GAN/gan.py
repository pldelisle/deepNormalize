import torch.nn as nn

from DeepNormalize.model.ResNet.ResNet import ResNet

from DeepNormalize.blocks.ResNet.bottleneck import Bottleneck
from DeepNormalize.blocks.ResNet.basic import BasicBlock


class Discriminator(nn.Module):

    def __init__(self, config, n_gpus):
        super(Discriminator, self).__init__()
        self.n_gpus = n_gpus
        self.model = ResNet(BasicBlock, [3, 4, 6, 3])

    def forward(self, x):
        if x.is_cuda and self.n_gpus > 1:
            output = nn.parallel.data_parallel(self.model, x, range(self.n_gpus))
        else:
            output = self.model(x)
        return output
