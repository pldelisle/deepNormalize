# -*- coding: utf-8 -*-
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
Simple utility functions.

"""

import nibabel as nib
import re
import os
import numpy as np


def pad_image_and_center(image):
    centered_image = np.zeros((240, 240, 160), dtype=np.int64)
    d = int(np.floor(np.abs(centered_image.shape[0] - image.shape[0])) / 2)
    e = int(np.floor(np.abs(centered_image.shape[1] - image.shape[1])) / 2)
    f = int(np.floor(np.abs(centered_image.shape[2] - image.shape[2])) / 2)
    centered_image[d:d + image.shape[0], e:e + image.shape[1], f:f + image.shape[2]] = image

    return centered_image


def load_image_obj(filename):
    try:
        image = nib.load(filename)
        image_data = image.get_data().astype(np.int64)
        return image_data
    except nib.filebasedimages.ImageFileError:
        raise IOError('Nibabel could not load image file: {}'.format(filename))


def _get_file_paths(path):
    file_paths = list()
    for dirpath, dirnames, filenames in os.walk(path):
        for name in filenames:
            f = os.path.join(dirpath, name)
            file_paths.append(f)

        file_paths = list(filter(None, file_paths))

    return file_paths


def find_files(path, modalities):
    files = list()

    for modality in modalities:
        modality = modality.lower()
        regex = re.compile(r"" + modality + r".nii")

        file_paths = _get_file_paths(path)

        modality_files = list(filter(regex.search, file_paths))

        files = files + modality_files

    return files


def find_segmentations(path):
    file_paths = _get_file_paths(path)
    regex_seg = re.compile(r"seg.nii")
    segmentation_files = list(filter(regex_seg.search, file_paths))

    return segmentation_files


def load_dataset(files):
    data_set = list()

    for file in files:
        image = load_image_obj(file)
        data_set.append(image)

    return data_set


