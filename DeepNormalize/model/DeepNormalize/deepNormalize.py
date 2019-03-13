from torch import nn

from DeepNormalize.model.UNet.unet import UNet


class DeepNormalize(nn.Module):

    def __init__(self, config, n_gpus):
        super(DeepNormalize, self).__init__()
        self.n_gpus = n_gpus

        self.unet_1 = UNet(config, in_channels=2, is_preprocessor=True)
        self.unet_2 = UNet(config, in_channels=2, is_preprocessor=False)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if x.is_cuda and self.n_gpus > 1:
            normalized_output = nn.parallel.data_parallel(self.unet_1, x, range(self.n_gpus))
            out = nn.parallel.data_parallel(self.unet_2, normalized_output, range(self.n_gpus))
        else:
            normalized_output = self.unet_1(x)
            out = self.unet_2(normalized_output)
        return normalized_output, out