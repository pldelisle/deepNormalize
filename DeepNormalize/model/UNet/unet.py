from torch import nn

from DeepNormalize.blocks.UNet.unet_block import UNetBlock
from DeepNormalize.layers.UNet.elementwise import ElementWise


class UNet(nn.Module):

    def __init__(self, config, in_channels, out_channels=2, is_preprocessor=False):
        super(UNet, self).__init__()

        # Define layer's hyperparameters.
        self.n_filters = config.get("n_filters")
        self.n_classes = config.get("n_classes")

        # Define activation function.
        self.activation_func = config.get("activation_func")

        self.down_layer_1 = UNetBlock("DOWNSAMPLE",
                                      (in_channels, self.n_filters[0]),
                                      (self.n_filters[0], self.n_filters[1]),
                                      (3, 3),
                                      activation_func=self.activation_func)

        self.down_layer_2 = UNetBlock("DOWNSAMPLE",
                                      (self.n_filters[1], self.n_filters[1]),
                                      (self.n_filters[1], self.n_filters[2]),
                                      (3, 3),
                                      activation_func=self.activation_func)

        self.down_layer_3 = UNetBlock("DOWNSAMPLE",
                                      (self.n_filters[2], self.n_filters[2]),
                                      (self.n_filters[2], self.n_filters[3]),
                                      (3, 3),
                                      activation_func=self.activation_func)

        self.up_layer_4 = UNetBlock("UPSAMPLE",
                                    (self.n_filters[3], self.n_filters[3]),
                                    (self.n_filters[3], self.n_filters[4]),
                                    (3, 3),
                                    activation_func=self.activation_func)

        self.concat_block_3 = ElementWise("CONCAT")

        self.up_layer_3 = UNetBlock("UPSAMPLE",
                                    (self.n_filters[3] + self.n_filters[4], self.n_filters[3]),
                                    (self.n_filters[3], self.n_filters[3]),
                                    (3, 3),
                                    activation_func=self.activation_func)

        self.concat_block_2 = ElementWise("CONCAT")

        self.up_layer_2 = UNetBlock("UPSAMPLE",
                                    (self.n_filters[2] + self.n_filters[3], self.n_filters[2]),
                                    (self.n_filters[2], self.n_filters[2]),
                                    (3, 3),
                                    activation_func=self.activation_func)

        self.concat_block_1 = ElementWise("CONCAT")

        self.up_layer_1 = UNetBlock("NONE",
                                    (self.n_filters[1] + self.n_filters[2], self.n_filters[1]),
                                    (self.n_filters[1], self.n_filters[1]),
                                    (3, 3),
                                    activation_func=self.activation_func)

        if is_preprocessor:
            self.out = UNetBlock("NONE",
                                 [self.n_filters[1]],
                                 [out_channels],
                                 [1],
                                 with_batch_norm=False,
                                 padding="valid",
                                 activation_func=None)
        else:
            self.out = UNetBlock("NONE",
                                 [self.n_filters[1]],
                                 [self.n_classes],
                                 [1],
                                 with_batch_norm=False,
                                 padding="valid",
                                 activation_func="softmax")

    def forward(self, x):
        pool_1, conv_1 = self.down_layer_1(x)

        pool_2, conv_2 = self.down_layer_2(pool_1)

        pool_3, conv_3 = self.down_layer_3(pool_2)

        up_3, _ = self.up_layer_4(pool_3)

        concat_3 = self.concat_block_3(conv_3, up_3)

        up_2, _ = self.up_layer_3(concat_3)

        concat_2 = self.concat_block_2(conv_2, up_2)

        up_1, _ = self.up_layer_2(concat_2)

        concat_1 = self.concat_block_1(conv_1, up_1)

        layer_1, _ = self.up_layer_1(concat_1)

        out, _ = self.out(layer_1)

        return out
