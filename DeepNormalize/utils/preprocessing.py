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
Utility function for preprocessing medical images.

"""

import numpy as np
from sklearn.feature_extraction.image import extract_patches

TOLERANCE = 1e-5
BACKGROUND_VALUE = 0


def preprocess_images(subject_modalities, apply):
    """
    Preprocess a list of images. Each first-layer list represents a modality, while
    each modality contains a list of "batch_size" images, each belonging to a subject.
    Modalities in order: FLAIR, T1, T1ce, T2, Segmentation, Weight Map.
    :param subject_modalities: A dictionary containing image modalities of one subject.
    :return: An dictionary containing preprocessed image modalities and slice template.
    """

    if apply:
        slices, modalities = crop(subject_modalities, apply)
        return slices, modalities

        # Additional preprocessing in pure Python/NumPy here.
        # ...

    else:
        slices = crop(subject_modalities, apply)
        return slices


def crop(subject_modalities, apply):
    """
    Crop a set of image modalities.
    :param subject_modalities:  A dictionary containing image modalities of one subject.
    :return: A dictionary containing the cropped images and slice template.
    """
    # Get the T1 modality slices which is in position 1.
    t1 = subject_modalities["t1"]

    # Template used for each subject is the T1 modality.
    t1_slices = get_slices(t1)

    if apply:
        # Create a new dictionary for storing crop result.
        modalities = dict()

        for key, value in subject_modalities.items():
            cropped_modality = value[tuple(t1_slices)]
            modalities[key] = cropped_modality

        return t1_slices, modalities

    else:
        return t1_slices


def get_slices(volume_data):
    """
    Return a set of slices to crop.
    :param volume_data: a Numpy array containing voxel values.
    :return: cropped_data : a Numpy array containing slices that defines the range of the crop.
    E.g. [slice(20, 200), slice(40, 150), slice(0, 100)] defines a 3D cube
    """
    # Logical OR to find non-background values. True if voxel value is below or above 0.
    idx = np.logical_or(volume_data < (BACKGROUND_VALUE - TOLERANCE),
                        volume_data > (BACKGROUND_VALUE + TOLERANCE))

    passes_threshold = np.any(idx, axis=-1)
    coords = np.array(np.where(passes_threshold))
    start = coords.min(axis=1)
    end = coords.max(axis=1) + 1

    # Pad with one voxel to avoid resampling problems.
    start = np.maximum(start - 1, 0)
    end = np.minimum(end + 1, volume_data.shape[:3])

    # Create slices of the volume.
    slices = [slice(s, e, 1) for s, e in zip(start, end)]

    return slices


def get_patches(subject_modalities, patch_shape, extraction_step):
    """
    Get a list of patches for each modality.
    :param subject_modalities: A dictionary containing image modalities of one subject.
    :param patch_shape: A 4-D Python list containing Height, Width, Depth size of patches ( [H, W, D, 1] ).
    :param extraction_step: integer or tuple of length arr.ndim. Indicates step size at which extraction shall
                            be performed. If integer is given, then the step is uniform in all dimensions.
    :return: A dictionary containing a list of patches for each image modality.
    """
    modalities = dict()

    for key, value in subject_modalities.items():
        # Use Scikit-Learn's method for extracting patches.
        patches = extract_patches(arr=value, patch_shape=patch_shape, extraction_step=extraction_step)
        # Reshape in a list of patches.
        patches = patches.reshape([-1] + list(patch_shape))
        # Assign modality of its list of patches.
        modalities[key] = patches

    return modalities
