import tensorflow as tf


class ElementWise(tf.keras.Model):
    def __init__(self, func):
        self.func = func
        super(ElementWise, self).__init__(name="")

        if self.func == "CONCAT":
            self.concat = tf.keras.layers.Concatenate(axis=1)
        else:
            raise NotImplementedError("Element wise function not implemented.")

    def call(self, input_tensor):
        return self.concat(input_tensor)
