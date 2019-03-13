from torch import nn

from DeepNormalize.layers.layer_util import expand_spatial_params


class DownSample(nn.Module):

    def __init__(self, func, kernel_size, stride):
        super(DownSample, self).__init__()

        self.func = func
        self.kernel_size = kernel_size
        self.stride = stride

        self.stride_all_dims = expand_spatial_params(self.stride, spatial_rank=3)
        self.kernel_size_all_dims = expand_spatial_params(self.kernel_size, spatial_rank=3)

        if self.func == "MAX":
            self.max_pooling = nn.MaxPool3d(kernel_size=self.kernel_size_all_dims,
                                            stride=self.stride_all_dims)

        else:
            raise NotImplementedError("Pooling operation not implemented.")

    def forward(self, x):
        return self.max_pooling(x)
