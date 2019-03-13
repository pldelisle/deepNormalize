from torch import nn


class Convolution3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, activation_func, with_bn, with_bias):
        super(Convolution3D, self).__init__()

        self.with_bn = with_bn

        if isinstance(padding, str):
            if padding == "replication":
                self.padding_layer = nn.ReplicationPad3d(padding=1)
                self.padding = 0
            elif padding == "valid":
                self.padding_layer = None
                self.padding = 0
            else:
                raise NotImplementedError("Padding layer not implemented.")

        elif isinstance(padding, int):
            self.padding = padding

        self.convolution = nn.Conv3d(in_channels=in_channels,
                                     out_channels=out_channels,
                                     kernel_size=kernel_size,
                                     padding=self.padding,
                                     stride=stride,
                                     bias=with_bias)

        if activation_func == "leaky_relu":
            self.activation_func = nn.LeakyReLU()
        elif activation_func == "relu":
            self.activation_func = nn.ReLU(inplace=True)
        elif activation_func == "softmax":
            self.activation_func = nn.Softmax(dim=1)
        elif activation_func is None:
            self.activation_func = None
        else:
            raise NotImplementedError("Activation function not implemented.")

        if with_bn:
            self.batch_norm = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        if self.padding_layer is not None:
            x = self.padding_layer(x)

        x = self.convolution(x)

        if self.activation_func is not None:
            x = self.activation_func(x)

        if self.with_bn:
            x = self.batch_norm(x)

        return x


class Deconvolution3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Deconvolution3D, self).__init__()
        self.padding = padding

        if self.padding == "replication":
            self.padding_layer = nn.ReplicationPad3d(padding=1)

        self.convolution = nn.ConvTranspose3d(in_channels=in_channels,
                                              out_channels=out_channels,
                                              kernel_size=kernel_size,
                                              padding=0,
                                              stride=stride, )

    def forward(self, x):
        # if self.padding == "replication":
        #     x = self.padding_layer(x)
        x = self.convolution(x)
        return x
