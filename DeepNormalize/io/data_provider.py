import tensorflow as tf
import os
import numpy as np

from DeepNormalize.utils import preprocessing


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
		self._batch_size = config.get("batch_size", 1)
		self._shuffle = config.get("shuffle", True)
		self._augment = config.get("augment", False)
		self._use_fp16 = config.get("fp16", False)
		self._patch_size = config.get("patch_size", 32)
		self._width = config.get("width", 240)
		self._height = config.get("height", 240)
		self._depth = config.get("depth", 155)
		self._n_channels = config.get("n_channels", 1)
		self._train_batch_size = config.get("train_batch_size", 32)

	# Declare a Sampler object here if required.

	def get_filename(self):
		return os.path.join(self._path, self._subset + ".tfrecords")

	@staticmethod
	def get_subjects_and_segmentations(path):
		"""
		Utility method to get, filter and arrange BraTS data set in a series of lists.
		:param path: The path to look for BraTS data set files.
		:return: A tuple containing multimodal MRI images for each subject and their respective segmentation.
		"""
		subjects = list()
		segs = list()

		for (dirpath, dirnames, filenames) in os.walk(path):
			if len(filenames) is not 0:
				segmentations = list(filter(lambda k: "seg.nii.gz" in k, filenames))
				files = filenames
				if segmentations:
					for seg in segmentations:
						files.remove(seg)
						segs.append(os.path.join(dirpath, seg))

				paths = list()
				for file in files:
					file = os.path.join(dirpath, file)
					paths.append(file)

				paths.sort()
				subjects.append(paths)

		del subjects[0]  # Remove the survival.csv file.

		return subjects, segs

	def _parser(self, serialized_example):
		"""Parses a single tf.Example which contains multiple modalities and label tensors."""

		# input format.
		features = tf.parse_single_example(
			serialized_example,
			features={
				"flair": tf.FixedLenFeature([self._width * self._height * self._depth * self._n_channels], tf.int64),
				"t1": tf.FixedLenFeature([self._width * self._height * self._depth * self._n_channels], tf.int64),
				"t1ce": tf.FixedLenFeature([self._width * self._height * self._depth * self._n_channels], tf.int64),
				"t2": tf.FixedLenFeature([self._width * self._height * self._depth * self._n_channels], tf.int64),
				"segmentation": tf.FixedLenFeature([self._width * self._height * self._depth * self._n_channels],
												   tf.int64),
				"weight_map": tf.FixedLenFeature([self._width * self._height * self._depth * self._n_channels],
												 tf.float32),
			})

		flair = features["flair"]
		t1 = features["t1"]
		t1ce = features["t1ce"]
		t2 = features["t2"]
		segmentation = features["segmentation"]
		weight_map = features["weight_map"]

		# Reshape from [depth * height * width] to [depth, height, width, channel].

		flair = tf.reshape(flair, [self._width, self._height, self._depth, self._n_channels])
		t1 = tf.reshape(t1, [self._width, self._height, self._depth, self._n_channels])
		t1ce = tf.reshape(t1ce, [self._width, self._height, self._depth, self._n_channels])
		t2 = tf.reshape(t2, [self._width, self._height, self._depth, self._n_channels])
		segmentation = tf.reshape(segmentation, [self._width, self._height, self._depth])
		weight_map = tf.reshape(weight_map, [self._width, self._height, self._depth])

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

		# Reshape from [depth * height * width] to [depth, height, width].
		segmentation = tf.cast(segmentation, dtype=tf.uint8)

		# Custom preprocessing here.

		return [flair, t1, t1ce, t2, segmentation, weight_map]

	def _read_py_function(self, flair, t1, t1ce, t2, segmentation, weight_map):
		"""
		A function in which previously extracted Tenors are casted to plain NumPy arrays for preprocessing.
		Preprocessing is done sequentially.
		:param flair: FLAIR modality of an image.
		:param t1: T1 modality of an image.
		:param t1ce: T1ce modality of an image.
		:param t2: T2 modality of an image.
		:param segmentation: Labels of an image.
		:param weight_map: Associated weight map of an image.
		:return: An array containing preprocessed modalities.
		"""

		# Call preprocessing facade method.
		flair, t1, t1ce, t2, segmentation, weight_map = preprocessing.preprocess_images(
			[flair, t1, t1ce, t2, segmentation, weight_map])

		# If necessary, call a sampler here.
		# patches, labels, weight_map = self._sampler.extract_class_balanced_samples(
		# 	[flair, t1, t1ce, t2, segmentation, weight_map])

		return flair, t1, t1ce, t2, segmentation, weight_map

	def crop_image(self, flair, t1, t1ce, t2, segmentation, weight_map):
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

		data = tf.stack([flair, t1, t1ce, t2, segmentation, weight_map], axis=-1)

		# Randomly crop a [self._patch_size, self._patch_size, self._patch_size] section of the image using
		# TensorFlow built-in function.
		image = tf.random_crop(data, [self._patch_size, self._patch_size, self._patch_size, 3])

		[flair, t1, t1ce, t2, segmentation, weight_map] = tf.unstack(image, 3, axis=-1)

		return flair, t1, t1ce, t2, segmentation, weight_map

	def input(self):

		filename = self.get_filename()

		with tf.name_scope("inputs"):
			# Instantiate a new data set based on a provided TFRecord file name.
			dataset = tf.data.TFRecordDataset(filename)

			# Repeat infinitely.
			dataset = dataset.repeat()

			# Parse records.
			dataset = dataset.map(map_func=self._parser, num_parallel_calls=self._batch_size)

			# Potentially shuffle records.
			if self._subset == "train" or "validation":
				min_queue_examples = int(
					DataProvider.num_examples_per_epoch(self._subset) * 0.10)
				# Ensure that the capacity is sufficiently large to provide good random
				# shuffling.
				dataset = dataset.shuffle(buffer_size=min_queue_examples + 2 * self._batch_size)

			# If using half precision float point (FP16), return type must be in tf.float16.
			if self._use_fp16:
				dataset = dataset.apply(tf.data.experimental.map_and_batch(
					map_func=lambda flair, t1, t1ce, t2, segmentation, weight_map: tf.py_func(
						self._read_py_function, [flair, t1, t1ce, t2, segmentation, weight_map],
						[tf.float16, tf.float16, tf.float16, tf.float16, tf.float16, tf.float16]
					), batch_size=self._batch_size, num_parallel_calls=self._batch_size, drop_remainder=True))

			else:  # If using single precision floating point.
				dataset = dataset.apply(tf.data.experimental.map_and_batch(
					map_func=lambda flair, t1, t1ce, t2, segmentation, weight_map: tf.py_func(
						self._read_py_function, [flair, t1, t1ce, t2, segmentation, weight_map],
						[tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32]
					), batch_size=self._batch_size, num_parallel_calls=self._batch_size, drop_remainder=True))

			if self._subset == "train":
				dataset = dataset.map(self.crop_image, num_parallel_calls=self._batch_size)

			# dataset = dataset.map(map_func=lambda flair, t1, t1ce, t2, segmentation, weight_map: tf.py_func(
			# 	self._read_py_function, [flair, t1, t1ce, t2, segmentation, weight_map],
			# 	[tf.float32, tf.float32, tf.float32, tf.float32, tf.uint8, tf.float32]
			# ))

			# Batch it up
			dataset = dataset.batch(self._train_batch_size)

			# Prepare for next iterations.
			dataset = dataset.prefetch(2 * self._batch_size)

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
