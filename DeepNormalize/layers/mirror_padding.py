import tensorflow as tf


class SymmetricPadding(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SymmetricPadding, self).__init__(**kwargs)

    def build(self, input_shape):
        # Make sure to call the `build` method at the end
        super(SymmetricPadding, self).build(input_shape)

    def call(self, inputs):
        self.x = tf.pad(inputs, paddings=([0, 0], [0, 0], [1, 1], [1, 1], [1, 1]), mode="SYMMETRIC")
        return self.x

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.output_dim
        return tf.TensorShape(shape)

    def get_config(self):
        base_config = super(SymmetricPadding, self).get_config()
        base_config['output_dim'] = self.output_dim
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
