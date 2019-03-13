import torch.nn as nn

from DeepNormalize.model.ResNet.ResNet import ResNet

from DeepNormalize.blocks.ResNet.bottleneck import Bottleneck
from DeepNormalize.blocks.ResNet.basic import BasicBlock


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = ResNet(BasicBlock, [3, 4, 6, 3])

    def forward(self, x):
        out = self.model(x)

        return out
