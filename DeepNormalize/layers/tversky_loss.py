import tensorflow as tf


class TverskyLoss(object):

    def __init__(self, n_classes, alpha, beta):
        self.n_classes = n_classes
        self.alpha = alpha
        self.beta = beta

    def to_one_hot(self, ground_truth):
        """
           Converts ground truth labels to one-hot, sparse tensors.
           Used extensively in segmentation losses.

           :param ground_truth: ground truth categorical labels (rank `N`)
           :param num_classes: A scalar defining the depth of the one hot dimension
               (see `depth` of `tf.one_hot`)
           :return: one-hot sparse tf tensor
               (rank `N+1`; new axis appended at the end)
           """
        # read input/output shapes
        if isinstance(self.n_classes, tf.Tensor):
            num_classes_tf = tf.to_int32(self.n_classes)
        else:
            num_classes_tf = tf.constant(self.n_classes, tf.int32)
        input_shape = tf.shape(tf.squeeze(ground_truth, axis=1))
        output_shape = tf.concat(
            [input_shape, tf.reshape(num_classes_tf, (1,))], 0)

        if self.n_classes == 1:
            return tf.reshape(ground_truth, output_shape)

        # squeeze the spatial shape
        ground_truth = tf.reshape(ground_truth, (-1,))
        # shape of squeezed output
        dense_shape = tf.stack([tf.shape(ground_truth)[0], num_classes_tf], 0)

        # create a rank-2 sparse tensor
        ground_truth = tf.to_int64(ground_truth)
        ids = tf.range(tf.to_int64(dense_shape[0]), dtype=tf.int64)
        ids = tf.stack([ids, ground_truth], axis=1)
        one_hot = tf.SparseTensor(
            indices=ids,
            values=tf.ones_like(ground_truth, dtype=tf.float32),
            dense_shape=tf.to_int64(dense_shape))

        # resume the spatial dims
        one_hot = tf.sparse_reshape(one_hot, output_shape)
        return one_hot

    def tversky(self, prediction, ground_truth, weight_map=None):
        """
          Function to calculate the Tversky loss for imbalanced data

              Sadegh et al. (2017)

              Tversky loss function for image segmentation
              using 3D fully convolutional deep networks

          :param prediction: the logits
          :param ground_truth: the segmentation ground_truth
          :param alpha: weight of false positives
          :param beta: weight of false negatives
          :param weight_map:
          :return: the loss
          """
        prediction = tf.cast(prediction, tf.float32)
        ground_truth = tf.cast(ground_truth, tf.int32)
        one_hot = tf.one_hot(tf.squeeze(ground_truth, axis=1), depth=self.n_classes, axis=1)

        p0 = prediction
        p1 = 1 - prediction
        g0 = one_hot
        g1 = 1 - one_hot

        if weight_map is not None:
            num_classes = prediction.shape[1].value
            weight_map_flattened = tf.reshape(weight_map, [-1])
            weight_map_expanded = tf.expand_dims(weight_map_flattened, 1)
            weight_map_nclasses = tf.tile(weight_map_expanded, [1, num_classes])
        else:
            weight_map_nclasses = 1

        tp = tf.reduce_sum(weight_map_nclasses * p0 * g0)
        fp = self.alpha * tf.reduce_sum(weight_map_nclasses * p0 * g1)
        fn = self.beta * tf.reduce_sum(weight_map_nclasses * p1 * g0)

        EPSILON = 0.00001
        numerator = tp
        denominator = tp + fp + fn + EPSILON
        score = numerator / denominator
        return 1.0 - tf.reduce_mean(score)
