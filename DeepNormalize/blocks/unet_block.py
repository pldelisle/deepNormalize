import tensorflow as tf

from DeepNormalize.layers.convolution import Convolution3D, Deconvolution3D
from DeepNormalize.layers.downsample import DownSample
from DeepNormalize.layers.layer_util import infer_spatial_rank


class UNetBlock(tf.keras.Model):
    def __init__(self,
                 func,
                 n_channels,
                 kernel_sizes,
                 padding="symmetric",
                 with_batch_norm=True,
                 activation_func=None):
        super(UNetBlock, self).__init__(name="")
        self.func = func
        self.kernel_sizes = kernel_sizes
        self.n_channels = n_channels
        self.padding = padding
        self.activation_func = activation_func
        self.with_batch_norm = with_batch_norm

    def call(self, input_tensor):
        x = input_tensor

        for kernel_size, filters in zip(self.kernel_sizes, self.n_channels):
            conv = Convolution3D(filters=filters,
                                 kernel_size=kernel_size,
                                 with_bn=self.with_batch_norm,
                                 padding=self.padding,
                                 activation_func=self.activation_func)
            x = conv(x)

        branch_output = x

        if self.func == "DOWNSAMPLE":
            downsample = DownSample("MAX",
                                    kernel_size=2,
                                    strides=2)
            x = downsample(x)

        elif self.func == "UPSAMPLE":
            upsample = Deconvolution3D(filters=self.n_channels[-1],
                                       kernel_size=2,
                                       strides=2)
            x = upsample(x)

        elif self.func == "NONE":
            pass

        else:
            raise NotImplementedError("UNet Block function not implemented.")

        return x, branch_output
