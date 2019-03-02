import tensorflow as tf

from DeepNormalize.layers.mirror_padding import SymmetricPadding
from DeepNormalize.layers.layer_util import expand_spatial_params


class Convolution3D(tf.keras.Model):
    def __init__(self, filters, kernel_size, with_bn, padding, activation_func):
        super(Convolution3D, self).__init__(name="")
        self.with_bn = with_bn
        self.padding = padding

        if self.padding == "symmetric":
            self.padding_layer = SymmetricPadding()

        self.convolution = tf.keras.layers.Convolution3D(filters=filters,
                                                         kernel_size=kernel_size,
                                                         padding="valid",
                                                         data_format="channels_first",
                                                         activation=activation_func,
                                                         use_bias=True,
                                                         kernel_initializer=tf.keras.initializers.glorot_uniform,
                                                         bias_initializer=tf.keras.initializers.Zeros,
                                                         kernel_regularizer=tf.keras.regularizers.L1L2(0.01, 0.01),
                                                         bias_regularizer=tf.keras.regularizers.L1L2(0.01, 0.01))
        if self.with_bn:
            self.batch_norm = tf.keras.layers.BatchNormalization()

    def call(self, input_tensor):
        x = input_tensor
        if self.padding == "symmetric":
            x = self.padding_layer(input_tensor)
        x = self.convolution(x)
        if self.with_bn:
            x = self.batch_norm(x)
        return x


class Deconvolution3D(tf.keras.Model):
    def __init__(self, filters, kernel_size, strides, activation_func=None):
        super(Deconvolution3D, self).__init__(name="")

        strides_all_dims = expand_spatial_params(strides, spatial_rank=3)

        self.padding = SymmetricPadding()
        self.convolution = tf.keras.layers.Conv3DTranspose(filters=filters,
                                                           kernel_size=kernel_size,
                                                           strides=strides_all_dims,
                                                           padding="same",
                                                           data_format="channels_first",
                                                           activation=activation_func,
                                                           use_bias=True,
                                                           kernel_initializer=tf.keras.initializers.glorot_uniform,
                                                           bias_initializer=tf.keras.initializers.Zeros,
                                                           kernel_regularizer=tf.keras.regularizers.L1L2(0.01, 0.01),
                                                           bias_regularizer=tf.keras.regularizers.L1L2(0.01, 0.01))

    def call(self, input_tensor):
        # x = self.padding(input_tensor)
        x = self.convolution(input_tensor)
        return x
