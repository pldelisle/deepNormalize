from torch import nn

from DeepNormalize.model.UNet.unet import UNet


class DeepNormalize(nn.Module):

    def __init__(self, config):
        super(DeepNormalize, self).__init__()

        self.unet_1 = UNet(config, in_channels=2, is_preprocessor=True)

        self.unet_2 = UNet(config, in_channels=2, is_preprocessor=False)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out_normalizer = self.unet_1(x)

        out = self.unet_2(out_normalizer)

        return out_normalizer, out