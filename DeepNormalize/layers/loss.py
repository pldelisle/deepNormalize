# -*- coding: utf-8 -*-
# Copyright 2019 Pierre-Luc Delisle. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
Different loss functions for a segmentation neural network.

"""

import tensorflow as tf


def labels_to_one_hot(ground_truth, num_classes=1):
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
    if isinstance(num_classes, tf.Tensor):
        num_classes_tf = tf.to_int32(num_classes)
    else:
        num_classes_tf = tf.constant(num_classes, tf.int32)
    input_shape = tf.shape(ground_truth)
    output_shape = tf.concat(
        [input_shape, tf.reshape(num_classes_tf, (1,))], 0)

    if num_classes == 1:
        # need a sparse representation?
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


def generalised_dice_loss(prediction,
                          ground_truth,
                          weight_map=None,
                          type_weight='Uniform'):
    """
    Function to calculate the Generalised Dice Loss defined in
        Sudre, C. et. al. (2017) Generalised Dice overlap as a deep learning
        loss function for highly unbalanced segmentations. DLMIA 2017
    :param prediction: the logits
    :param ground_truth: the segmentation ground truth
    :param weight_map:
    :param type_weight: type of weighting allowed between labels (choice
        between Square (square of inverse of volume),
        Simple (inverse of volume) and Uniform (no weighting))
    :return: the loss
    """
    prediction = tf.cast(prediction, tf.float32)
    if len(ground_truth.shape) == len(prediction.shape):
        ground_truth = ground_truth[..., -1]
    one_hot = labels_to_one_hot(ground_truth, tf.shape(prediction)[-1])

    if weight_map is not None:
        num_classes = prediction.shape[1].value
        # weight_map_nclasses = tf.reshape(
        #     tf.tile(weight_map, [num_classes]), prediction.get_shape())
        weight_map_nclasses = tf.tile(
            tf.expand_dims(tf.reshape(weight_map, [-1]), 1), [1, num_classes])
        ref_vol = tf.sparse_reduce_sum(
            weight_map_nclasses * one_hot, reduction_axes=[0])

        intersect = tf.sparse_reduce_sum(
            weight_map_nclasses * one_hot * prediction, reduction_axes=[0])
        seg_vol = tf.reduce_sum(
            tf.multiply(weight_map_nclasses, prediction), 0)
    else:
        ref_vol = tf.sparse_reduce_sum(one_hot, reduction_axes=[0])
        intersect = tf.sparse_reduce_sum(one_hot * prediction,
                                         reduction_axes=[0])
        seg_vol = tf.reduce_sum(prediction, 0)
    if type_weight == 'Square':
        weights = tf.reciprocal(tf.square(ref_vol))
    elif type_weight == 'Simple':
        weights = tf.reciprocal(ref_vol)
    elif type_weight == 'Uniform':
        weights = tf.ones_like(ref_vol)
    else:
        raise ValueError("The variable type_weight \"{}\""
                         "is not defined.".format(type_weight))
    new_weights = tf.where(tf.is_inf(weights), tf.zeros_like(weights), weights)
    weights = tf.where(tf.is_inf(weights), tf.ones_like(weights) *
                       tf.reduce_max(new_weights), weights)
    generalised_dice_numerator = \
        2 * tf.reduce_sum(tf.multiply(weights, intersect))
    # generalised_dice_denominator = \
    #     tf.reduce_sum(tf.multiply(weights, seg_vol + ref_vol)) + 1e-6
    generalised_dice_denominator = tf.reduce_sum(
        tf.multiply(weights, tf.maximum(seg_vol + ref_vol, 1)))
    generalised_dice_score = \
        generalised_dice_numerator / generalised_dice_denominator
    generalised_dice_score = tf.where(tf.is_nan(generalised_dice_score), 1.0,
                                      generalised_dice_score)
    return 1 - generalised_dice_score


def dice_coefficient(prediction, ground_truth, smooth=1.0):
    """
    Function to calculate the dice coefficient.
    :param prediction: the logits (before softmax)
    :param ground_truth: the segmentation ground truth
    :param smooth: Smoothing coefficient.
    :return: the dice coefficient.
    """

    y_true_f = tf.layers.flatten(ground_truth)
    y_pred_f = tf.layers.flatten(prediction)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)


def dice_coefficient_loss(prediction, ground_truth):
    """
     Function to calculate the dice loss.
     :param prediction: the logits (before softmax)
     :param ground_truth: the segmentation ground truth
     :return: the dice loss.
     """
    return -dice_coefficient(prediction, ground_truth)


def weighted_cross_entropy(prediction, ground_truth, weight_map=None):
    """
    Function to calculate the weighted cross-entropy loss.
    :param prediction: the logits (before softmax)
    :param ground_truth: the segmentation ground truth
    :param weight_map: A list of weights for each classes.
    :return: the dice loss.
    """
    entropy = tf.nn.weighted_cross_entropy_with_logits(
        logits=prediction, targets=ground_truth, pos_weight=weight_map)

    return tf.reduce_mean(entropy)


def cross_entropy(prediction, ground_truth, weight_map=None):
    """
    Function to calculate the cross-entropy loss function
    :param prediction: the logits (before softmax)
    :param ground_truth: the segmentation ground truth
    :param weight_map: A 3D weight map of shape "predictions".
    :return: the cross-entropy loss
    """
    # if len(ground_truth.shape) == len(prediction.shape):
    #     ground_truth = ground_truth[..., -1]

    # # TODO trace this back:
    # ground_truth = tf.cast(ground_truth, tf.int32)

    entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=prediction, labels=ground_truth)

    if weight_map is None:
        return tf.reduce_mean(entropy)

    weight_sum = tf.maximum(tf.reduce_sum(weight_map), 1e-6)
    return tf.reduce_sum(entropy * weight_map / weight_sum)


def tversky(prediction, ground_truth, weight_map=None, alpha=0.5, beta=0.5):
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
    prediction = tf.to_float(prediction)
    if len(ground_truth.shape) == len(prediction.shape):
        ground_truth = ground_truth[..., -1]
    one_hot = labels_to_one_hot(ground_truth, tf.shape(prediction)[-1])
    one_hot = tf.sparse.to_dense(one_hot)

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
    fp = alpha * tf.reduce_sum(weight_map_nclasses * p0 * g1)
    fn = beta * tf.reduce_sum(weight_map_nclasses * p1 * g0)

    EPSILON = 0.00001
    numerator = tp
    denominator = tp + fp + fn + EPSILON
    score = numerator / denominator
    return 1.0 - tf.reduce_mean(score)
