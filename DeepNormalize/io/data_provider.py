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
A class for providing data to a TensorFlow neural network.

"""

import tensorflow as tf
import os


class DataProvider(object):

    def __init__(self, path, subset, config):
        """
        Data provider for multimodal MRI.
        :param tfrecord_path: path to the TFRecords
        :param subset: the subset (train, validation, test) the data set will be instantiate with.
        :param config: A JSON configuration file for hyper parameters.
        """

        self._path = path
        self._subset = subset
        self._subject_batch_size = config.get("subject_batch_size", 1)
        self._shuffle = config.get("shuffle", True)
        self._use_fp16 = config.get("use_fp16", False)
        self._patch_shape = config.get("volume")["patch_size"]
        self._width = config.get("width", 240)
        self._height = config.get("height", 240)
        self._depth = config.get("depth", 155)
        self._n_channels = config.get("n_channels", 1)
        self._batch_size = config.get("batch_size", 32)
        self._n_modalities = config.get("n_modalities", 4)
        self._use_weight_map = config.get("use_weight_map", True)

        if self._use_weight_map:
            self._n_total_modalities = self._n_modalities + 2
        else:
            self._n_total_modalities = self._n_modalities + 1

        # Declare a Sampler object here if required.

    def get_filename(self):
        """
        Returns the file name associated to the data set's subset.
        :return: A string containing the complete path and file name.
        """
        return os.path.join(self._path, self._subset + ".tfrecords")

    def _training_parser(self, serialized_example):
        """
        Parses a single tf.Example which contains multiple modalities and label tensors.
        :param serialized_example: A TFRecord serialized example.
        :return: A tuple containing all parsed image modalities, segmentation and weight_map with slice dimensions.
        """

        # input format.
        features = tf.parse_single_example(
            serialized_example,
            features={
                "flair": tf.FixedLenFeature(self._width * self._height * self._depth * self._n_channels,
                                            tf.int64),
                "t1": tf.FixedLenFeature(self._width * self._height * self._depth * self._n_channels,
                                         tf.int64),
                "t1ce": tf.FixedLenFeature(self._width * self._height * self._depth * self._n_channels,
                                           tf.int64),
                "t2": tf.FixedLenFeature(self._width * self._height * self._depth * self._n_channels,
                                         tf.int64),
                "segmentation": tf.FixedLenFeature(self._width * self._height * self._depth * self._n_channels,
                                                   tf.int64),
                "weight_map": tf.FixedLenFeature(self._width * self._height * self._depth * self._n_channels,
                                                 tf.float32),
                "original_shape": tf.FixedLenFeature(4,
                                                     tf.int64),
                "slices": tf.FixedLenFeature(9,
                                             tf.int64),

            })

        flair = features["flair"]
        t1 = features["t1"]
        t1ce = features["t1ce"]
        t2 = features["t2"]
        segmentation = features["segmentation"]
        weight_map = features["weight_map"]
        original_shape = features["original_shape"]
        slices = features["slices"]

        # Reshape from [depth * height * width] to [depth, height, width, channel] using serialized image shape.
        flair = tf.reshape(flair, original_shape)
        t1 = tf.reshape(t1, original_shape)
        t1ce = tf.reshape(t1ce, original_shape)
        t2 = tf.reshape(t2, original_shape)
        segmentation = tf.reshape(segmentation, original_shape)
        weight_map = tf.reshape(weight_map, original_shape)

        if self._use_fp16:
            flair = tf.cast(flair, dtype=tf.float16)
            t1 = tf.cast(t1, dtype=tf.float16)
            t1ce = tf.cast(t1ce, dtype=tf.float16)
            t2 = tf.cast(t2, dtype=tf.float16)
            segmentation = tf.cast(segmentation, dtype=tf.float16)
            weight_map = tf.cast(weight_map, dtype=tf.float16)

        else:
            flair = tf.cast(flair, dtype=tf.float32)
            t1 = tf.cast(t1, dtype=tf.float32)
            t1ce = tf.cast(t1ce, dtype=tf.float32)
            t2 = tf.cast(t2, dtype=tf.float32)
            segmentation = tf.cast(segmentation, dtype=tf.float32)
            weight_map = tf.cast(weight_map, dtype=tf.float32)

        return flair, t1, t1ce, t2, segmentation, weight_map, slices

    def crop_image(self, flair, t1, t1ce, t2, segmentation, weight_map, slices):
        """
        Crop modalities.
        :param flair: FLAIR modality of an image.
        :param t1: T1 modality of an image.
        :param t1ce: T1ce modality of an image.
        :param t2: T2 modality of an image.
        :param segmentation: Labels of an image.
        :param weight_map: Associated weight map of an image.
        :return: Randomly cropped 3D patches from image's set.
        """

        stack = tf.stack([flair, t1, t1ce, t2, segmentation, weight_map], axis=-1)

        stack = stack[slices[0]:slices[2]:slices[1], slices[3]:slices[5]:slices[4], slices[6]:slices[8]:slices[7], :, :]

        # Randomly crop a [self._patch_size, self._patch_size, self._patch_size, 1] section of the image using
        # TensorFlow built-in function.
        image = tf.random_crop(stack, [self._patch_shape, self._patch_shape, self._patch_shape, self._n_channels,
                                       self._n_total_modalities])

        [flair, t1, t1ce, t2, segmentation, weight_map] = tf.unstack(image, self._n_total_modalities, axis=-1)

        return flair, t1, t1ce, t2, tf.cast(segmentation, tf.uint8), weight_map

    def input(self):
        """
        Return a TensorFlow data set object containing inputs.
        :return: dataset: TensorFlow data set object.
        """

        # Get file names for this data set.
        filename = self.get_filename()

        with tf.name_scope("inputs"):
            # Instantiate a new data set based on a provided TFRecord file name. Generate a Dataset with raw records.
            dataset = tf.data.TFRecordDataset(filename)

            # Parse records. Returns 1 volume with all its modalities.
            dataset = dataset.map(map_func=self._training_parser,
                                  num_parallel_calls=1)

            # Prefetch 10 subject.
            dataset = dataset.prefetch(10)

            if self._subset == "train" or "validation":
                min_queue_examples = int(
                    DataProvider.num_examples_per_epoch(self._subset) * 0.10)

                # Ensure that the capacity is sufficiently large to provide good random
                # shuffling.
                dataset = dataset.shuffle(buffer_size=min_queue_examples + 2 * self._batch_size)

                # Repeat indefinitely.
                dataset = dataset.repeat()

                # Returns randomly cropped images.
                dataset = dataset.map(self.crop_image, num_parallel_calls=self._batch_size)

            # Batch 3D patches. Here, we want (training batch / number of modalities) samples per modality,
            # which is usually 32 / 4 = 8 patches per modality.
            dataset = dataset.batch(int(self._batch_size / self._n_modalities))

            # Prepare for next iterations.
            dataset = dataset.prefetch(10 * self._batch_size)

            return dataset

    @staticmethod
    def num_examples_per_epoch(subset):
        if subset == "train":
            # Return the number of volumes in training dataset.
            return int(230)
        if subset == "validation":
            # Return the number of volumes in validation dataset.
            return int(29)
        if subset == "test":
            # Return the number of volumes in test dataset.
            return int(26)
