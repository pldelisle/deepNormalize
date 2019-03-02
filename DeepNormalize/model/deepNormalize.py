import tensorflow as tf

from DeepNormalize.blocks.unet_block import UNetBlock
from DeepNormalize.layers.elementwise import ElementWise


class DeepNormalize(tf.keras.Model):

    def __init__(self, config, n_gpus):
        super(DeepNormalize, self).__init__()

        self.inputs = [tf.keras.Input(shape=(config.get("inputs")["n_modalities"],
                                             config.get("inputs")["patch_size"],
                                             config.get("inputs")["patch_size"],
                                             config.get("inputs")["patch_size"]),
                                      batch_size=config.get("inputs")["batch_size"])]

        self.n_filters = config.get("model")["n_filters"]
        self.n_classes = config.get("model")[("n_classes")]

        activation_func = config.get("model")["activation_func"]

        self.output_names = ["output_{}".format(i) for i in n_gpus]

        if activation_func == "relu":
            self.activation_func = tf.keras.activations.relu

        else:
            raise NotImplementedError("Activation function not implemented.")

        self.down_layer_1 = UNetBlock("DOWNSAMPLE",
                                      (self.n_filters[0], self.n_filters[1]),
                                      (3, 3),
                                      activation_func=self.activation_func)

        self.down_layer_2 = UNetBlock("DOWNSAMPLE",
                                      (self.n_filters[1], self.n_filters[2]),
                                      (3, 3),
                                      activation_func=self.activation_func)

        self.down_layer_3 = UNetBlock("DOWNSAMPLE",
                                      (self.n_filters[2], self.n_filters[3]),
                                      (3, 3),
                                      activation_func=self.activation_func)

        self.up_layer_4 = UNetBlock("UPSAMPLE",
                                    (self.n_filters[3], self.n_filters[4]),
                                    (3, 3),
                                    activation_func=self.activation_func)

        self.concat_block_3 = ElementWise("CONCAT")

        self.up_layer_2 = UNetBlock("UPSAMPLE",
                                    (self.n_filters[3], self.n_filters[3]),
                                    (3, 3),
                                    activation_func=self.activation_func)

        self.concat_block_2 = ElementWise("CONCAT")

        self.up_layer_1 = UNetBlock("UPSAMPLE",
                                    (self.n_filters[2], self.n_filters[2]),
                                    (3, 3),
                                    activation_func=self.activation_func)

        self.concat_block_1 = ElementWise("CONCAT")

        self.final_layer_block = UNetBlock("NONE",
                                           (self.n_filters[1], self.n_filters[1]),
                                           (3, 3),
                                           activation_func=self.activation_func)

        self.preprocessor_out = UNetBlock("NONE",
                                          [self.n_classes],
                                          [1],
                                          with_batch_norm=False,
                                          padding="valid",
                                          activation_func=tf.keras.activations.softmax)

        self.outputs = [self.preprocessor_out]

    def call(self, inputs, training=True):
        pool_1, conv_1 = self.down_layer_1(inputs[0])

        pool_2, conv_2 = self.down_layer_2(pool_1)

        pool_3, conv_3 = self.down_layer_3(pool_2)

        up_3, _ = self.up_layer_4(pool_3)

        concat_3 = self.concat_block_3([conv_3, up_3])

        up_2, _ = self.up_layer_2(concat_3)

        concat_2 = self.concat_block_2([conv_2, up_2])

        up_1, _ = self.up_layer_1(concat_2)

        concat_1 = self.concat_block_1([conv_1, up_1])

        layer_1, _ = self.final_layer_block(concat_1)

        out, _ = self.preprocessor_out(layer_1)

        return out
