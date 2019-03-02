import tensorflow as tf

from DeepNormalize.layers.layer_util import expand_spatial_params, infer_spatial_rank


class DownSample(tf.keras.Model):

    def __init__(self, func, kernel_size, strides):
        super(DownSample, self).__init__(name="")

        self.func = func
        self.kernel_size = kernel_size
        self.strides = strides

        stride_all_dims = expand_spatial_params(self.strides, spatial_rank=3)
        kernel_size_all_dims = expand_spatial_params(self.kernel_size, spatial_rank=3)
        if self.func == "MAX":
            self.max_pooling = tf.keras.layers.MaxPool3D(pool_size=kernel_size_all_dims,
                                                         strides=stride_all_dims,
                                                         data_format="channels_first")
        else:
            raise NotImplementedError("Pooling operation not implemented.")

    def call(self, input_tensor):
        return self.max_pooling(input_tensor)
