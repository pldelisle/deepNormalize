#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from torch import nn

from DeepNormalize.layers.UNet.convolution import Convolution3D, Deconvolution3D
from DeepNormalize.layers.UNet.downsample import DownSample


class UNetBlock(nn.Module):
    def __init__(self,
                 func,
                 in_channels,
                 out_channels,
                 kernel_sizes,
                 padding="replication",
                 with_batch_norm=True,
                 activation_func=None):
        super(UNetBlock, self).__init__()
        self.func = func
        self.kernel_sizes = kernel_sizes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding
        self.activation_func = activation_func
        self.with_batch_norm = with_batch_norm

        self.conv = nn.ModuleList()

        for (kernel_size, in_channel, out_channel) in (zip(self.kernel_sizes, self.in_channels, self.out_channels)):
            self.conv.append(
                Convolution3D(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=1,
                              padding=self.padding, activation_func=self.activation_func, with_bn=self.with_batch_norm,
                              with_bias=True))
        if self.func == "DOWNSAMPLE":
            self.downsample = DownSample("MAX",
                                         kernel_size=2,
                                         stride=2)
        elif self.func == "UPSAMPLE":
            self.upsample = Deconvolution3D(in_channels=self.out_channels[-1],
                                            out_channels=self.out_channels[-1],
                                            kernel_size=2,
                                            stride=2,
                                            padding="replication")

        elif self.func == "NONE":
            pass

        else:
            raise NotImplementedError("UNet Block function not implemented.")

    def forward(self, x):

        for layer in self.conv:
            x = layer(x)

        branch_output = x

        if self.func == "DOWNSAMPLE":

            x = self.downsample(x)

        elif self.func == "UPSAMPLE":

            x = self.upsample(x)

        elif self.func == "NONE":
            pass

        else:
            raise NotImplementedError("UNet Block function not implemented.")

        return x, branch_output
