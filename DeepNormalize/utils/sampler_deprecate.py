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
A Sampler object for obtaining balanced inputs. Unused.

"""

import numpy as np


class Sampler(object):

	def __init__(self, patch_size, batch_size, n_classes, use_weight_map):
		"""
		This object samples balanced patches from a subject's image's list.
		Subject's segmentation (ground truth) is assumed to be subject[4] position in array.
		If a weight map is present, the latter is assumed to be in subject[5] position in array.
		:param patch_size: The desired 3D patch size (3D array list).
		:param batch_size: The number of 3D patch to extract for 1 subject (int).
		:param n_classes: The number of classes in the data set (int).
		:param use_weight_map: Whether the problem uses a weight map modality or not (bool).
		"""
		self._patch_size = patch_size
		self._batch_size = batch_size
		self._n_classes = n_classes
		self._use_weight_map = use_weight_map

	def extract_class_balanced_samples(self, subject):
		"""
		Extract a class-balanced set of samples from a subject.
		:param subject: A list containing subject's image modalities, segmentation and weight map.
		:return:
		"""

		# Re-arrange n_classes in tuple.
		if isinstance(self._n_classes, int):
			self._n_classes = tuple(range(self._n_classes))
		n_classes = len(self._n_classes)

		rank = len(self._patch_size)

		# Find the number of elements to extract for each class.
		n_ex_per_class = np.ones(n_classes).astype(int) * int(np.round(self._batch_size / n_classes))

		# Compute an example radius to define the region to extract around a center location
		ex_rad = np.array(list(zip(np.floor(np.array(self._patch_size) / 2.0),
								   np.ceil(np.array(self._patch_size) / 2.0))),
						  dtype=np.int32)

		# Loop over the image modalities.
		for m in range(n_classes):
			current_modality = subject[m]

			class_ex_images = []
			class_ex_lbls = []
			class_ex_wms = []
			min_ratio = 1.

			# Loop over each classes.
			for c_idx, c in enumerate(self._n_classes):
				idx = np.argwhere(subject[4] == c)

				ex_images = []
				ex_lbls = []
				ex_wms = []

				if len(idx) == 0 or n_ex_per_class[c_idx] == 0:
					class_ex_images.append([])
					class_ex_lbls.append([])
					class_ex_wms.append([])
					continue

				# Extract random locations.
				r_idx_idx = np.random.choice(len(idx),
											 size=min(n_ex_per_class[c_idx], len(idx)),
											 replace=False).astype(int)
				r_idx = idx[r_idx_idx]

				# Shift the random to valid locations if necessary.
				r_idx = np.array(
					[np.array([max(min(r[dim], current_modality.shape[dim] - ex_rad[dim][1]),
								   ex_rad[dim][0]) for dim in range(rank)])
					 for r in r_idx])

				# Loop over the random locations.
				for i in range(len(r_idx)):
					# Extract class-balanced examples from the original image
					slicer = [slice(r_idx[i][dim] -
									ex_rad[dim][0], r_idx[i][dim] +
									ex_rad[dim][1]) for dim in range(rank)]

					ex_image = current_modality[slicer][np.newaxis, :]

					ex_lbl = subject[4][slicer][np.newaxis, :]

					if self._use_weight_map:
						ex_wm = subject[5][slicer][np.newaxis, :]
						ex_wms = np.concatenate((ex_wms, ex_wm), axis=0) \
							if (len(ex_wms) != 0) else ex_wm

					# Concatenate them and return the examples
					ex_images = np.concatenate((ex_images, ex_image), axis=0) \
						if (len(ex_images) != 0) else ex_image
					ex_lbls = np.concatenate((ex_lbls, ex_lbl), axis=0) \
						if (len(ex_lbls) != 0) else ex_lbl

				class_ex_images.append(ex_images)
				class_ex_lbls.append(ex_lbls)

				if self._use_weight_map:
					class_ex_wms.append(ex_wms)

				ratio = n_ex_per_class[c_idx] / len(ex_images)
				min_ratio = ratio if ratio < min_ratio else min_ratio

			indices = np.floor(n_ex_per_class * min_ratio).astype(int)

			ex_images = np.concatenate([cimage[:idxs] for cimage, idxs in zip(class_ex_images, indices)
										if len(cimage) > 0], axis=0)
			ex_lbls = np.concatenate([clbl[:idxs] for clbl, idxs in zip(class_ex_lbls, indices)
									  if len(clbl) > 0], axis=0)

			if self._use_weight_map:
				ex_wms = np.concatenate([cwm[:idxs] for cwm, idxs in zip(class_ex_wms, indices)
										 if len(cwm) > 0], axis=0)

				return ex_images, ex_lbls, ex_wms
			else:
				return ex_images, ex_lbls
